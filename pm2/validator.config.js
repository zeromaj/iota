module.exports = {
    apps: [{
        name: 'iota-validator',
        script: './start_validators.sh',
        interpreter: 'bash',
        cwd: './',
        watch: false,
        instances: 1,
        autorestart: true,
        max_restarts: 10,
        env: {
            NODE_ENV: 'production',
        },
        error_file: './logs/validator-error.log',
        out_file: './logs/validator-out.log',
        time: true,
    }]
};
