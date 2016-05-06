var app = angular.module('andersonApp', ['ui.bootstrap']);

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
});

app.filter('tagFilter', function () {
    return function (input) {
        return input.join(', ')
    }
});

app.factory('tools', ['$http', function ($http) {

    /**
     * [min, max]
     * max is inclusive
     */
    var randint = function (max, min) {
        if (min == undefined) {
            min = 0;
        }
        return Math.floor(Math.random() * (max - min + 1) + min)
    };

    var get_bible_statement = function (callback) {
        var url = '/assets/anderson/bible.csv';
        return $http.get(url).then(function (response) {
            var statements = response.data.split('\n');
            statements = statements.filter(function (value) {
                return value.trim().length > 0;
            });

            var random_int = randint(statements.length - 1);
            callback(statements[random_int]);
        })
    };

    var get_fast_categories = function (callback) {
        var url = '/assets/anderson/fast_categories.csv';
        return $http.get(url).then(function (response) {
            var data = response.data.split('\n');
            data = data.filter(function (value) {
                return value.trim().length > 0;
            });
            callback(data);
        })
    };


    return {
        randint: randint,
        get_bible_statement: get_bible_statement,
        get_fast_categories: get_fast_categories
    }
}]);

app.controller('AndersonPostContoller', ['$scope', 'tools', function ($scope, tools) {
    // Set Posts
    $scope.posts = global_post_data;

    // Changing Search Text
    $scope.change_search = function (text) {
        $scope.searchText = text;
    };

    // Set Good Bible Statement
    tools.get_bible_statement(function (statement) {
        $scope.bible_statement = statement;
    });

    // Set Fast Categories
    tools.get_fast_categories(function (data) {
        $scope.fast_categories = data;
    });

}]);