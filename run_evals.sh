# arguments are a list of model names
# example: ./run_evals.sh model1 model2 model3
# if no arguments, then use default list of models

if [ $# -eq 0 ]
then
    echo "No arguments supplied, using default list of models"
    models="huggyllama/llama-7b bjoernp/trampeltier-7b huggyllama/llama-13b huggyllama/llama-30b togethercomputer/RedPajama-INCITE-7B-Base togethercomputer/RedPajama-INCITE-Base-3B-v1"
else
    models=$@
fi

for model in $models;
do
    # replace / with _ in model name
    model_escaped=${model//\//_}
    echo "Evaluating $model. Writing to ${model_escaped}_all.json and log to ${model_escaped}_all.log"
    srun -C gpu --gpus=1 --account laion ./eval/bin/python eval_de.py --model="hf-causal" --limit=50 --model_args="pretrained=${model},dtype=float16" --output_path="./${model_escaped}_all.json" --device="cuda" 2>&1 | tee "./${model_escaped}_all.log"
done