/**
 * Created by anderson on 15. 11. 5.
 */
var http = require('http');
var faye = require('faye');

var server = http.createServer();
var bayeux = new faye.NodeAdapter({mount: '/faye', timeout: 45});


bayeux.attach(server);
server.listen(8000);