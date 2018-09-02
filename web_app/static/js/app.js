function plot_confusion_matrix() {
    var path_to_trace = '/static/data/data.json'
    Plotly.d3.json(path_to_trace, function(error, response) {
        username = 'meguloabebe'
        api_key = 'HpYqtIl2jc639TTDnq7y'
        if (error) return console.log(error);
        var trace = response['trace']
        var data = response['data_list']
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
            title: 'Accuracy: Training & Validation',
            xaxis: {
                title:'Num. of Epochs',
                autorange: true
            },
            yaxis: {
                title:'Accuracy (%)',
                autorange: true
            }
        }

        // Plot Confusion Matrix
        var confusion_matrix_trace = [trace]
        var confusion_matrix_data = {
            annotations: [data]
        }

        console.log(data)
        Plotly.plot('confusion_matrix', {data:confusion_matrix_trace, layout:confusion_matrix_data})
        Plotly.plot('training_validation_accuracy', {data: train_val_acc_data, layout:train_val_acc_layout})
    })
}

plot_confusion_matrix()