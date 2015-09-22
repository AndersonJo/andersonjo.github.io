var app = angular.module('andersonApp', []);

app.config([
    '$interpolateProvider', function ($interpolateProvider) {
        return $interpolateProvider.startSymbol('{(').endSymbol(')}');
    }
]);

app.filter('andersonDate', function () {
    return function (input) {
        var date = new Date(input);
        var month = date.getMonth();
        if(month == 0){
            month = 'Jan';
        }
        else if(month == 1){
            month = 'Feb';
        }
        else if(month == 2){
            month = 'Mar';
        }
        else if(month == 3){
            month = 'Apr';
        }
        else if(month == 4){
            month = 'May';
        }
        else if(month == 5){
            month = 'Jun';
        }
        else if(month == 6){
            month = 'Jul';
        }
        else if(month == 7){
            month = 'Aug';
        }
        else if(month == 8){
            month = 'Sep';
        }
        else if(month == 9){
            month = 'Oct';
        }
        else if(month == 10){
            month = 'Nov';
        }
        else if(month == 11){
            month = 'Dec';
        }
        return  month +' '+ date.getDate() +' '+ date.getFullYear() +' ';
    }
})

app.controller('AndersonPostContoller', ['$scope', function ($scope) {
    $scope.posts = global_post_data;

    $scope.search_text
}]);