# Automated Puzzle Solver
Some helpful how to on using the automated solver.
## GUI Navigation
You can move around the puzzle using the arrow keys, the on screen buttons, or dragging with the mouse. The image can be rotated with the rotation buttons. Zooming can be done with the zoom buttons or by using the scrollwheel on the mouses. The settings menu is found by clicking on the gear.
## Instructions 
1. To start the solver GUI
    * `cd src`
    * `python solverWindow.py`
2. In the Upload pieces section, select the + button to begin uploading pieces. 
   For convenience, we have included JSON files which will load all the required puzzle info into the system.
   In the file section, select the JSON file of choice. Not all of the given puzzles run successfully, but we thought
   we would include them all anyways. This is because some puzzles have bad information.
   For good puzzles, try beachHut.JSON (300 pc), donut.JSON (1000 pc), pokemonBeach.JSON (500 pc), tart.JSON (300 pc), owl.JSON (300 pc)
   
   
   For an example of how to use without the JSON files, try the tart puzzle using the table below for an input file with its given piece count
      +--------------------------+---+\
      | input/tart_puzzle_01.jpg |30 |\
      +--------------------------+---+\
      | input/tart_puzzle_02.jpg |30  |\
      +--------------------------+---+\
      | input/tart_puzzle_03.jpg |30  |\
      +--------------------------+---+\
      | input/tart_puzzle_04.jpg |30  |\
      +--------------------------+---+\
      | input/tart_puzzle_05.jpg |30  |\
      +--------------------------+---+\
      | input/tart_puzzle_06.jpg | 30 |\
      +--------------------------+---+\
      | input/tart_puzzle_07.jpg |28  |\
      +--------------------------+---+\
      | input/tart_puzzle_08.jpg |30  |\
      +--------------------------+---+\
      | input/tart_puzzle_09.jpg |30  |\
      +--------------------------+---+\
      | input/tart_puzzle_10.jpg |30  |\
      +--------------------------+---+\
      | input/tart_puzzle_11.jpg |26  |\
      +--------------------------+---+
3. Set the settings for the puzzle (skip if using a JSON file)
    * For tart set dimension to 18 18
    * Generation defaults are Size 100, with 100 generations
4. Click run 
