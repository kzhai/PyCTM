PyCTM
==========

PyCTM is a Correlated Topic Modeling package, please download the latest version from our [GitHub repository](https://github.com/kzhai/PyCTM).

Please send any bugs of problems to Ke Zhai (kzhai@umd.edu).

Install and Build
----------

This package depends on many external python libraries, such as numpy, scipy and nltk.

Launch and Execute
----------

Assume the PyCTM package is downloaded under directory ```$PROJECT_SPACE/src/```, i.e.,

	$PROJECT_SPACE/src/PyCTM

To prepare the example dataset,

	tar zxvf nips-abstract.tar.gz

To launch PyCTM, first redirect to the directory of PyCTM source code,

	cd $PROJECT_SPACE/src/PyCTM

and run the following command on example dataset,

	python -m launch_train --input_directory=./nips-abstract --output_directory=./ --number_of_topics=10 --training_iterations=50

The generic argument to run PyCTM is

	python -m launch_train --input_directory=$INPUT_DIRECTORY/$CORPUS_NAME --output_directory=$OUTPUT_DIRECTORY --number_of_topics=$NUMBER_OF_TOPICS --training_iterations=$NUMBER_OF_ITERATIONS

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$CORPUS_NAME```.

Under any circumstances, you may also get help information and usage hints by running the following command

	python -m launch_train --help
	