var trajectory_json;
var trajectory_length;

function updateGrid(flat_grid, score) {
    this.score = score
    $("#score").text("Score: " + this.score);
    $('.cell-value').each(function(cell, obj) {
        let cell_value = flat_grid[cell];
        if (cell_value == 0) {
            $(this).text("");
            $(this).parent().css({'background-color': '#e6f0ff'});
        } else {
            $(this).text(flat_grid[cell]);
            let color_lightness = 50 / (Math.log2(flat_grid[cell]) + 1) + 50;
            let hsl_color = 'hsl(255, 100%, ' + color_lightness.toString() + '%)';
            $(this).parent().css({'background-color': hsl_color});
        }
    })
}

$(document).ready(function() {

    let intervalId = false;
    let flat_grid = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let done = false;
    let game_score = 0;

    $.getJSON("../example/trajectory_0.json", function(json){trajectory_json = json;});

    $('#run').click(() => {
        run()
    });

    function run() {
        done = false;
        this.game_score = 0
        trajectory_length = trajectory_json['actions'].length;
        let step = 0;
        if (intervalId) {
            clearInterval(intervalId);
            intervalId = false;
        }
        intervalId = setInterval(function(){
            if (done) {
                clearInterval(intervalId);
                intervalId = false;
                $("#status").text("Game Over");
                return;
            }
            if (step == 0) {
                updateGrid(trajectory_json['states'][step], 0);
            } else if (0 < step && step < trajectory_length) {
                console.log('active')
                updateGrid(trajectory_json['states'][step], trajectory_json['scores'][step-1]);
            } else if (step == trajectory_length) {
                updateGrid(trajectory_json['terminal_state'], trajectory_json['scores'][step-1]);
                done = true;
            }
            step++;
        }, 100)
    }

})