var assert = require('assert'),
    faye = require('faye');


describe('Faye', function () {
    describe('Test Connection', function () {
        it('should return connection', function (done) {
            var client = new faye.Client('http://localhost:8000');
            console.log(client);

            done()

        })
    })
});