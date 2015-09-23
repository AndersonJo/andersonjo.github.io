var app = angular.module('andersonApp', []);

app.config([
    '$interpolateProvider', function ($interpolateProvider) {
        return $interpolateProvider.startSymbol('{(').endSymbol(')}');
    }
]);

app.filter('andersonDate', function () {
    var month_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return function (input) {
        var date = new Date(input);
        var month = date.getMonth();
        month = month_abbr[month];
        return month + ' ' + date.getDate() + ' ' + date.getFullYear() + ' ';
    }
})

app.controller('AndersonPostContoller', ['$scope', function ($scope) {
    $scope.posts = global_post_data;

}]);