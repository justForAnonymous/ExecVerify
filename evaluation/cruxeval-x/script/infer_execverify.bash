python /cruxeval-x/inference/infer_exec_verify_concurrency.py \
    --langs "['java','cpp', 'cs','go','js','php']" \
    --model_name /path/to/your/model \
    --model_dir /cruxeval-x/model  \
    --data_root /cruxeval-x/datasets/cruxeval-x \
    --data_input_output /cruxeval-x/datasets/cruxeval_preprocessed \
    --example_root /cruxeval-x/datasets/examples \
    --example_input_output /cruxeval-x/datasets/examples_preprocessed \
    --output_dir /cruxeval-x/infer_results