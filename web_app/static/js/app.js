function plot_confusion_matrix() {
    var path_to_trace = '/static/data/trace.json'
    Plotly.d3.json(path_to_trace, function(error, response) {
        if (error) return console.warn(error);
    })
}