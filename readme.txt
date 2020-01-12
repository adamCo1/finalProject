
 Yuval Zehavi - 204251102
 Yuval Elbaz - 204386858
 Adam Cohen - 301914388

this is a prototype for a genetic algorithm .

as we were directed to do, we implemented an algorithm that uses Mutual Information as fitness function.
the target of our project is NOT to use this algorithm, it is only used for being able to see an algorithm that works
and as a proof of concept.

the input of the algorithm can be seen in the GUI. the overall population size must be greater then other population parameters.

** insert only numbers. when inserting invalid parameters a value error will be raised with the error 'please enter valid parameters'.

this prototype runs on the wdbc.csv which is attached to the program. please select it as input. this database is for classifying
breast cancer.

** the dataset calculations is being done on 10 columns. the names printed are the names of the columns as they are
in the dataset.

output : retrieves a list of the best features. as we use Mutual Information as fitness function we want to see all of the features
in that list.