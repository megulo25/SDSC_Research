function plot_confusion_matrix() {
    var path_to_trace = '/static/data/data.json'
    Plotly.d3.json(path_to_trace, function(error, response) {
        if (error) return console.log(error);
        var trace = response['trace']
        var data = response['data_list']

        console.log(trace)
    })
}

plot_confusion_matrix()