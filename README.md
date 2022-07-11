# Expected threat (xT) model based on Wyscout data
These Python modules implement an interpretation of the Expected Threat (xT) model as described by [Karun Sing (2018)](https://karun.in/blog/expected-threat.html).
Except for just implementing a working code for creating and running the models, emphasis has been put on making the application "modular" in a way which enables better readability (e.g clearer representation of the formula in the code), changeability (e.g paramater configuration) and analyzeability (e.g see the underlying calculations and values of the model outputs).

This project has been implemented for learning purposes and as first step for me to engage and contribute to the Football analytics community. I'm open to any feedback on this implementation!

## Model details
The xT model is computed in the form of a grid with X*Y cells representing the football pitch where each event feeding the model occurs at a position (X,Y) on the grid.

The xT for each cell is computed according to the following formula:
![xT formula by Karun Singh](xT_formula.png)

## Data
Wyscout event data from 2017/2018 major leagues and Fifa World Cup 2018 made available through the research of [Pappalardo et al., (2019)](https://doi.org/10.6084/m9.figshare.c.4415000.v5).

The Wyscout pitch event coordinate system:
![The Wyscout pitch coordinate system](WyscoutPitchCoordinates.png)

## Running the project
1. Download the following data sets from [this site](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5):
  - events
  - matches
  - players
  - teams
2. Place the data files (json) in a new folder named wyscoutdata on the same level as the root folder (wyscout-xt-model) as indicated by the `load_*_data()` functions the in `wyscoutmodelutlity.py module`, or just change to any other path that your prefer.
3. Run the entire `modelcreator.py` to produce all the outputs or run indivudual sections (cells) of the script to produce desired outputs.
4. Play around by changing different model parameter values (UPPERCASE_VARIABLES)

## References
- Pappalardo, Luca; Massucco, Emanuele (2019): Soccer match event dataset. figshare. Collection. https://doi.org/10.6084/m9.figshare.c.4415000.v5
- Karun Sing (2018): Introducing Expected Threat (xT). https://karun.in/blog/expected-threat.html
