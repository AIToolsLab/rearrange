module.exports = {
    devServer: {
        port: 5001,
        public: 'https://dev1.kenarnold.org/',
        allowedHosts: [
            '.kenarnold.org'
        ],
        proxy: {
            '/api': {
                target: 'http://localhost:5009'
            }
        }
    }
};