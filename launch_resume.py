import pickle;
import optparse
import string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

import numpy;
import shutil

# model_settings_pattern = re.compile('\d+-\d+-ctm_inferencer-I(?P<iteration>\d+)-S(?P<snapshot>\d+)-aa(?P<alpha>[\d\.]+)(-smh(?P<smh>[\d]+))?(-sp(?P<sp>[\d]+)-mp(?P<mp>[\d]+))?');
model_settings_pattern = re.compile('\d+-\d+-ctm-I(?P<iteration>\d+)-S(?P<snapshot>\d+)-K(?P<topic>\d+)-am(?P<alpha_mu>[\d\.]+)-as(?P<alpha_sigma>[\d\.]+)-ab(?P<alpha_beta>[\d\.]+)');

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        # input_file=None,
                        model_directory=None,
                        snapshot_index=-1,

                        # parameter set 2
                        output_directory=None,
                        training_iterations=-1,
                        snapshot_interval=-1,
                        )
    # parameter set 1
    # parser.add_option("--input_file", type="string", dest="input_file",
                      # help="input directory [None]");
    # parser.add_option("--input_directory", type="string", dest="input_directory",
                      # help="input directory [None]");
    parser.add_option("--model_directory", type="string", dest="model_directory",
                      help="model directory [None]");
    parser.add_option("--snapshot_index", type="int", dest="snapshot_index",
                      help="snapshot index [-1]");
    # parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      # help="number of training iterations [1000]");
    # parser.add_option("--dataset_name", type="string", dest="dataset_name",
                      # help="the corpus name [None]");

    # parameter set 2
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    # parser.add_option("--alpha_alpha", type="float", dest="alpha_alpha",
                      # help="hyper-parameter for Dirichlet process of cluster [1]")
    # parser.add_option("--alpha_kappa", type="float", dest="alpha_kappa",
                      # help="hyper-parameter for top level Dirichlet process of distribution over topics [1]")
    # parser.add_option("--alpha_nu", type="float", dest="alpha_nu",
                      # help="hyper-parameter for bottom level Dirichlet process of distribution over topics [1]")
    parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      help="number of training iterations [-1]");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [-1 (default): remain unchanged]");
                      
    (options, args) = parser.parse_args();
    return options;
    
def main():
    options = parse_args();
    
    assert(options.model_directory != None);
    model_directory = options.model_directory;
    
    if not os.path.exists(model_directory):
        sys.stderr.write("model directory %s not exists...\n" % (model_directory));
        return;
    model_directory = model_directory.rstrip("/");
    model_settings = os.path.basename(model_directory);
    
    assert options.snapshot_index > 0
    snapshot_index = options.snapshot_index;
    
    # load the existing model
    model_snapshot_file_path = os.path.join(model_directory, "model-%d" % snapshot_index);
    if not os.path.exists(model_snapshot_file_path):
        sys.stderr.write("error: model snapshot file unfound %s...\n" % (model_snapshot_file_path));
        return;
    
    ctm_inferencer = pickle.load(open(model_snapshot_file_path, "rb"));
    print('successfully load model snapshot %s...' % (os.path.join(model_directory, "model-%d" % snapshot_index)));

    # set the resume options  
    matches = re.match(model_settings_pattern, model_settings);
    
    # training_iterations = int(matches.group('iteration'));
    training_iterations = options.training_iterations;
    assert training_iterations > snapshot_index;
    if options.snapshot_interval == -1:
        snapshot_interval = int(matches.group('snapshot'));
    else:
        snapshot_interval = options.snapshot_interval;
    number_of_topics = int(matches.group('topic'));
    alpha_mu = float(matches.group('alpha_mu'));
    alpha_sigma = float(matches.group('alpha_sigma'));
    alpha_beta = float(matches.group('alpha_beta'));
    
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    suffix += "-%s" % ("ctm");
    suffix += "-I%d" % (training_iterations);
    suffix += "-S%d" % (snapshot_interval);
    suffix += "-K%g" % (number_of_topics);
    suffix += "-am%g" % (alpha_mu);
    suffix += "-as%g" % (alpha_sigma);
    suffix += "-ab%g" % (alpha_beta);

    assert options.output_directory != None;
    output_directory = options.output_directory;
    output_directory = output_directory.rstrip("/");
    output_directory = os.path.join(output_directory, suffix);
    assert (not os.path.exists(os.path.abspath(output_directory)));
    os.mkdir(os.path.abspath(output_directory));
    
    shutil.copy(model_snapshot_file_path, os.path.join(output_directory, "model-" + str(snapshot_index)));
    shutil.copy(model_snapshot_file_path, os.path.join(output_directory, "exp_beta-" + str(snapshot_index)));
    
    for iteration in range(snapshot_index, training_iterations):
        # clock = time.time();
        log_likelihood = ctm_inferencer.learning();
        # clock = time.time()-clock;
        # print 'training iteration %d finished in %f seconds: number-of-clusters = %d, log-likelihood = %f' % (dpgm._iteration_counter, clock, dpgm._K, log_likelihood);
        
        if ((ctm_inferencer._counter) % snapshot_interval == 0):
            ctm_inferencer.export_beta(os.path.join(output_directory, 'exp_beta-' + str(ctm_inferencer._counter)));
            model_snapshot_path = os.path.join(output_directory, 'model-' + str(ctm_inferencer._counter));
            pickle.dump(ctm_inferencer, open(model_snapshot_path, 'wb'));
    
    model_snapshot_path = os.path.join(output_directory, 'model-' + str(ctm_inferencer._counter));
    pickle.dump(ctm_inferencer, open(model_snapshot_path, 'wb'));
    
if __name__ == '__main__':
    main()
