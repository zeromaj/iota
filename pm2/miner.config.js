module.exports = {
    apps: [{
        name: 'iota-miner',
        script: './start_miner.sh',
        interpreter: 'bash',
        cwd: './',
        watch: false,
        instances: 1,
        autorestart: true,
        max_restarts: 10,
        env: {
            NODE_ENV: 'production',
        },
        error_file: './logs/miner-error.log',
        out_file: './logs/miner-out.log',
        time: true,
    }]
};
