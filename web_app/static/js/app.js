function plot_confusion_matrix() {
    var path_to_trace = '/static/data/data.json'
    Plotly.d3.json(path_to_trace, function(error, response) {
        username = 'meguloabebe'
        api_key = 'HpYqtIl2jc639TTDnq7y'
        if (error) return console.log(error);
        var trace_data = response['trace_data']
        var trace_layout = response['trace_layout']
        var train_acc = response['train_acc']
        var val_acc = response['val_acc']

        var n = [...Array(val_acc.length).keys()]

        // Plot Acc: Training vs. Validation
        var train_acc_trace = {
            x:n,
            y:train_acc,
            type:'scatter',
            name:'Training'
        };

        var val_acc_trace = {
            x:n,
            y:val_acc,
            type:'scatter',
            name:'Validation'
        };

        var train_val_acc_data = [train_acc_trace, val_acc_trace];
        var train_val_acc_layout = {
            xaxis: {
                title:'Num. of Epochs',
                autorange: true,
            },
            yaxis: {
                title:'Accuracy (%)',
                autorange: true,
            }
        }

        // Plot Confusion Matrix
        margin = {
            "margin": {
                "l":150
            }
        }
        console.log(trace_layout)
        trace_layout = Object.assign({}, trace_layout, margin)
        console.log(trace_layout)
        Plotly.plot('confusion_matrix', {data: [trace_data], layout:trace_layout})
        Plotly.plot('training_validation_accuracy', {data: train_val_acc_data, layout:train_val_acc_layout})
    })
}

plot_confusion_matrix()