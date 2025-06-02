# S3 Multipart Upload Implementation

This document describes the orchestrator-coordinated multipart upload functionality implemented to handle large weight files in the swarm MVP system.

## Problem

The original implementation used S3's single-part upload via presigned POST URLs, which has a 5GB file size limit. When uploading large neural network weights (especially for large language models), this limit can be exceeded, causing upload failures.

## Solution

We've implemented S3 multipart upload using orchestrator-coordinated presigned URLs to handle large files. The system automatically:

1. **Miners request upload coordination** from the orchestrator with file size information
2. **Orchestrator decides upload method** based on file size and generates appropriate presigned URLs
3. **Uses single-part upload** for files smaller than the configured threshold via regular presigned POST URLs
4. **Uses multipart upload** for larger files with orchestrator-managed sessions
5. **Uploads parts concurrently** using presigned PUT URLs for better performance
6. **Completes upload via orchestrator** which manages the S3 multipart session

## Architecture

### Flow Overview

```
Miner → Orchestrator → S3
  ↓         ↓          ↓
Request → Generate → Upload
Upload    Presigned   Parts
Info      URLs
  ↓         ↓          ↓
Upload → Monitor →   Complete
Parts    Progress    Upload
```

### Components

1. **Orchestrator Storage API** (`storage/api.py`)
   - `/storage/multipart_upload/initiate` - Start multipart upload session
   - `/storage/multipart_upload/complete` - Complete multipart upload
   - `/storage/multipart_upload/abort` - Abort failed multipart upload

2. **API Client** (`miner/api_client.py`)
   - `initiate_multipart_upload()` - Request multipart upload from orchestrator
   - `complete_multipart_upload()` - Complete upload via orchestrator
   - `abort_multipart_upload()` - Abort upload via orchestrator

3. **Smart Upload Function** (`utils/s3_interactions.py`)
   - `smart_upload_via_orchestrator_async()` - Orchestrator-coordinated smart upload

4. **Miner Integration** (`miner/miner.py`)
   - `upload_weights()` - Upload model weights using orchestrator coordination
   - `upload_activation()` - Upload activations using orchestrator coordination

## Configuration

The system uses the following thresholds:

- **Minimum part size**: 5MB (AWS S3 requirement)
- **Maximum single-part size**: 5GB (AWS S3 limit)
- **Default part size**: 100MB (configurable)
- **Maximum concurrent parts**: 5 (configurable for bandwidth management)

## Security Model

### Access Control

- **Miners cannot generate presigned URLs directly** - all URL generation goes through orchestrator
- **Orchestrator validates entity permissions** using Epistula signatures
- **Time-limited presigned URLs** with configurable expiration (default 1 hour)
- **Path validation** ensures miners can only upload to their designated paths

### Authentication Flow

1. Miner signs request with wallet hotkey
2. Orchestrator validates Epistula signature
3. Orchestrator checks metagraph for miner registration
4. Orchestrator generates presigned URLs for validated paths only
5. Miner uses presigned URLs for direct S3 upload
6. Orchestrator manages multipart session lifecycle

## Usage Examples

### Basic Weight Upload (Miner)

```python
# In miner code - automatically uses orchestrator coordination
await miner.upload_weights(
    data=weight_tensor,
    miner_hotkey=miner.hotkey,
    num_sections=1,
    epoch=0
)
```

### Manual Smart Upload

```python
from utils.s3_interactions import smart_upload_via_orchestrator_async

# Convert tensor to bytes
data_bytes = tensor.view(torch.uint8).cpu().detach().numpy().tobytes()

# Upload via orchestrator coordination
s3_path = await smart_upload_via_orchestrator_async(
    api_client=miner.api_client,
    data=data_bytes,
    path="weights/miner_hotkey/1/0/weights.pt",
    part_size=50 * 1024 * 1024  # 50MB parts
)
```

### API Client Usage

```python
# Initiate multipart upload
upload_info = await api_client.initiate_multipart_upload(
    path="weights/example.pt",
    file_size=len(data_bytes),
    part_size=100 * 1024 * 1024
)

if upload_info["use_multipart"]:
    # Handle multipart upload
    multipart_data = upload_info["multipart_data"]
    upload_id = multipart_data["upload_id"]
    presigned_urls = multipart_data["presigned_urls"]

    # Upload parts using presigned URLs...

    # Complete upload
    await api_client.complete_multipart_upload(
        path="weights/example.pt",
        upload_id=upload_id,
        parts=completed_parts
    )
else:
    # Handle single-part upload
    presigned_data = upload_info["presigned_data"]
    # Upload using regular presigned POST...
```

## Testing

Run the test suite:

```bash
python -m pytest tests/test_s3_interactions.py -v
```

Run the demo script:

```bash
python example_multipart_upload.py
```

## Error Handling

The system includes comprehensive error handling:

- **Automatic abort on failure** - Failed multipart uploads are automatically aborted
- **Retry logic** - Network failures are retried with exponential backoff
- **Graceful degradation** - Falls back to single-part upload when appropriate
- **Cleanup on error** - Partial uploads are cleaned up to avoid S3 storage costs

## Performance

### Benefits

- **No file size limits** - Can handle files larger than 5GB
- **Concurrent uploads** - Multiple parts upload simultaneously
- **Better reliability** - Individual part failures don't fail entire upload
- **Cost optimization** - Failed uploads are automatically cleaned up

### Considerations

- **Minimum overhead** - Small files still use efficient single-part uploads
- **Bandwidth management** - Concurrent uploads are limited to prevent overwhelming networks
- **Memory efficient** - Large files are streamed in chunks rather than loaded entirely

## Migration Notes

- **Backward compatibility** - Existing single-part uploads continue to work
- **Automatic detection** - System automatically chooses best upload method
- **No API changes** - Miner upload methods remain the same
- **Enhanced security** - All uploads now go through orchestrator validation

## Monitoring

The orchestrator logs multipart upload events for monitoring:

- Upload initiation with file size and part count
- Upload completion with timing information
- Upload failures with error details
- Cleanup operations for failed uploads

## Troubleshooting

### Common Issues

1. **"Upload failed" errors** - Check network connectivity and S3 permissions
2. **"Multipart upload not found"** - Upload may have expired, retry from beginning
3. **"Access denied"** - Verify miner is registered and wallet is valid
4. **"File too large"** - System should handle automatically, check orchestrator logs

### Debug Mode

Enable debug logging to see detailed multipart upload progress:

```python
import logging
logging.getLogger("utils.s3_interactions").setLevel(logging.DEBUG)
```
