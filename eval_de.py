import argparse
import json
import logging
import os

from lm_eval import tasks, evaluator, utils
import lm_eval.models
import numpy as np
import pandas as pd

logging.getLogger("openai").setLevel(logging.WARNING)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default="output.json")
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default="./results.csv")
    parser.add_argument("--skip_fewshots", type=list, default=[])

    return parser.parse_args()

tasks_per_fewshot = {
    5: [
        "hendrycksTest*",
        "MMLU-DE*",
    ],
    10: [
        "hellaswag",
        "hellaswag_de"
    ],
    25: [
        "arc_challenge",
        "arc_challenge_de"
    ],
    0: [
        "truthfulqa_mc",
        "truthful_qa_de",
        "pawsx_de",
        "xnli_de",
        "lambada_openai_mt_de",
        # "wmt20-en-de",
    ]
}


def main():
    args = parse_args()
    all_results = {
        "results": {},
        "versions": {},
    }
    for num_fewshots, task_list in tasks_per_fewshot.items():
        task_names = utils.pattern_match(task_list, tasks.ALL_TASKS)
        results = evaluator.simple_evaluate(
            model=args.model,
            model_args=args.model_args,
            tasks=task_names,
            num_fewshot=num_fewshots,
            batch_size=args.batch_size,
            device=args.device,
            no_cache=args.no_cache,
            limit=args.limit,
            description_dict=None,
            decontamination_ngrams_path=args.decontamination_ngrams_path,
            check_integrity=args.check_integrity,
            write_out=args.write_out,
            output_base_path=args.output_base_path,
            bootstrap_iters=500
        )
        all_results["results"].update(results["results"])
        all_results["versions"].update(results["versions"])
        all_results["config"] = results["config"]
        dumped = json.dumps(all_results, indent=2)
        print(dumped)
        output_path = args.output_path.replace(".json", f"_{num_fewshots}shots.json")
        with open(output_path, "w") as f:
            f.write(dumped)

    all_results["config"]["model"] = args.model
    all_results["config"]["model_args"] = args.model_args
    mmlu_de_mean = np.mean([v["acc"] for k, v in all_results["results"].items() if "MMLU-DE" in k])
    mmlu_de_std_mean = np.mean([v["acc_stderr"] for k, v in all_results["results"].items() if "MMLU-DE" in k])
    mmlu_en_mean = np.mean([v["acc"] for k, v in all_results["results"].items() if "hendrycksTest" in k])
    mmlu_en_std_mean = np.mean([v["acc_stderr"] for k, v in all_results["results"].items() if "hendrycksTest" in k])
    all_results["results"]["MMLU-DE"] = {
        "acc": mmlu_de_mean,
        "acc_stderr": mmlu_de_std_mean
    }
    all_results["versions"]["MMLU-DE"] = all_results["versions"]["MMLU-DE-abstract_algebra"] # choose one subject to get version
    all_results["results"]["hendrycksTest"] = {
        "acc": mmlu_en_mean,
        "acc_stderr": mmlu_en_std_mean,
    }
    all_results["versions"]["hendrycksTest"] = all_results["versions"]["hendrycksTest-abstract_algebra"] # choose one subject to get version

    dumped = json.dumps(all_results, indent=2)
    print(dumped)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    # remove subtasks from results
    for k in list(all_results["results"].keys()):
        if "MMLU-DE-" in k or "hendrycksTest-" in k:
            del all_results["results"][k]
    print(evaluator.make_table(all_results))

    if os.path.exists(args.csv_path):
        df = pd.read_csv(args.csv_path)
    else:
        # metric can be acc, ppl, mc1, mc2, etc.
        columns = ["model", "model_args", "metric"] + list(all_results["results"].keys())
        df = pd.DataFrame(columns=columns)

    results = pd.DataFrame.from_dict(all_results["results"])
    results['model'] = args.model
    results['model_args'] = args.model_args
    results['metric'] = results.index
    results.reset_index(drop=True, inplace=True)
    # reorder columns
    cols = results.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    results = results[cols]
    # add current results to existing dataframe
    df = pd.concat([df, results], ignore_index=True).reset_index(drop=True)
    # remove duplicates, keeping the most recently added
    df.drop_duplicates(subset=['model', 'model_args', 'metric'], keep='last', inplace=True)
    # save to csv
    df.to_csv(args.csv_path, index=False)



if __name__ == "__main__":
    main()
