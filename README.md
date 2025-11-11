
# INR-ASSR-Emperical-Analysis

A centralized repository for training and evaluting existing INR-based ASSR implementaiotn, with the ablity to train using multiple-training recipes, evaluting using different IQA metrics, datasets, and encoders.


Running the training:
Got to the train.py, uncomment the training strategy you want to train the model for.
Running the testing:
Go to the test.py, uncommnet the testing strategy you want to test the model you want to.

Configurations:
There are several configurable items that can be changed to get extended benefits form the reporsitory. The current configured pipelines and models are sufficient to reproduce the results for the previous techniques under different settings for training and evalutions.
But the repository is menat to exapnd with ease to try out new training and evalution, datasets, INR-based encoder and decoders.

Rather than relying o external config files the configuraiotns are part of the python files.

Configrations:
The configuration classes in the Configuration module, hsa the required elements that can be configured for training, and validation.


Training Orchestrator:
Is the base for training using the different training configuraiotns. For any given encoder and decoder it cna be used to train for th efollowing configured training recipes.
TrainSimple: Training using the simplest training strategy using L1-Loss, 48^2 input patch size, scale range of 1-4, and multi-step based learning rate schedular.

TrainSimpleLargerPatch: Training using L1-Loss, 64^2 input patch size, scale range of 1-4, and multi-step based learning rate schedular.

TrainSimpleLargerPatchScale: Training using L1-Loss, 64^2 input patch size, scale range of 1-6, and multi-step based learning rate schedular.

TrainSGDR: Training using L1-Loss, 48^2 input patch size, scale range of 1-4, and Cosine Annealing learning rate based learning rate schedular.

TrainGradLoss: Training using Gradient-L1-Loss, 48^2 input patch size, scale range of 1-4, and multi-step based learning rate schedular.

TrainGramL1Loss: Training using Gram-L1-Loss, 48^2 input patch size, scale range of 1-4, and multi-step based learning rate schedular.

Adding ne wrecipes is simple, just copy paste the existing configuration and make the required combination you want.


Testing Orchestrator;
Is the base for testing using different strategies for testing the input model.
It has a few configuraitons that cna be manged, although most default are the ones that covers all the required cases.

BaseTestingConfiguration: file controls the testing configuraitons, with edfault values set for it already and specific to dataset being updated inside the individual orchestrator. 

{encoder}: Name of encoder (Models.EncoderType)
{decoder}: Name of decoder (Models.DecoderType)
{recipe}: Name of training recipe (Models.RecipeType)
{input_patch}: Input patch size
{scale_range}: Scale range used ofr training the model
{eval_scale}: Scale to test the data for (Orchestrator s iterating over differnet scales chaning this paramter dynamically)
{benchmark}: Type of the dataset being used (Used for determining if RGB or Y-channel based evalutions to be performed on a dataset) (Models.BenchmarkType)
{valid_data_path}: Path to the input dataset HR images
{valid_data_pathScale}: Path to the downsampled images, or keep it none and the code will auto-downsample and generate input images using the HR images, using bicubic downsampling
{total_example}: Total examples in the dataset for logging purposes
{eval_batch_size}: Size of coordinates to be sampled at a single time for input to the model.
{model_name}: To specify if to test the model with best PNSR, or the one trained for all the epochs (Models.SavedModelType)
{test_strategy}: To specifiy if the model should divide images into patches or not for more efficient testing on smaller GPUs (TestingStrategy)
{breakdown_patch_size}: In case patch based testing then specify the size of the patch, else leave it None
{overlap}: In case overalpping-patch based testing then specify the size of the patch, else leave it None

Individual settings will make up the model path to be laoded based on the specified training strategies so use the same settings as you used for training the model, and whose checkpoint you want to use to run tour tests. 

Again, this is configurable and can be configured for additionla settings, like selection of a specific epoch checkpoint, using different testing strageis like the patched and overlapped patched testing in case you have GPU emmeory contraints etc.


The default path for saving and loading the model weights is always gonna be of the following format:

./model_states_{encoder}_{decoder}_{recipe}_Patch_{input-patch}_Scale_{scale-range-start}_{scale-range-end}_{save-type}

Where anyhting in the braces is variable.
{encoder}: Name of encoder (Models.EncoderType)
{decoder}: Name of decoder (Models.DecoderType)
{recipe}: Name of training recipe (Models.RecipeType)
{input-patch}: Input patch size
{scale-range-start}: Scale range item 0
{scale-range-end}: Scale range item 1
{save-type}: Model checkpoint (Models.SavedModelType, or the number of epoch for which the state was saved)

But you can always customize it in the Utitlies.PathManager.GetModelSavePath



