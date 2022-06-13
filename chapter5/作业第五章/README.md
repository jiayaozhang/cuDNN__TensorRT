# BERT Python Demo

## Setting Up Your Environment

1.  Create and launch the docker image
    ```
    sh python/create_docker_container.sh
    ```

3. Build the plugins and download the fine-tuned models
    ```
    cd TensorRT/demo/BERT && sh python/build_examples.sh
    ```

## Building an Engine
To build an engine, run the `bert_builder.py` script. For example,
```
python python/bert_builder.py -m /workspace/models/fine-tuned/bert_tf_v2_base_fp16_384_v2/model.ckpt-8144 -o bert_base_384.engine -b 1 -s 384 -c /workspace/models/fine-tuned/bert_tf_v2_base_fp16_384_v2
```
This will build and engine with a maximum batch size of 1 (`-b 1`), and sequence length of 384 (`-s 384`) using the `bert_config.json` file located in `workspace/models/fine-tuned/bert_tf_v2_base_fp16_384_v2`

## Running Inference
Finally, you can run inference with the engine generated from the previous step using the `bert_inference.py` script. 

## BERT Inference with TensorRT

Refer to the Python script bert_inference.py and the detailed Jupyter notebook BERT_TRT.ipynb in the sample folder for a step-by-step description and walkthrough of the inference process. Let’s review a few key parameters and concepts to perform inference with TensorRT in this section.

BERT (more specifically the Encoder layer) uses the following parameters to govern its operation:

    Batch size
    Sequence Length
    Number of attention heads

The value of these parameters, which depend on the BERT model chosen, are used to set the configuration parameters for the TensorRT plan file (execution engine).

For each encoder, also specify the number of hidden layers and the attention head size. You can also read all the above parameters from the Tensorflow checkpoint file.

As the BERT model we are using has been fine-tuned for a downstream task of Question Answering on the SQuAD dataset, the output for the network (i.e. the output fully connected layer) will be a span of text where the answer appears in the passage (referred to as  h_output in the sample).  Once we generate the TensorRT engine, we can serialize it and use it later with TensorRT runtime.

During inference, we perform memory copies from CPU to GPU and vice versa asynchronously to get tensors into and out of the GPU memory, respectively.  Asynchronous memory copy operation hides latency of memory transfer by overlapping computations with memory copy operation between device and host. 

`https://developer.nvidia.com/blog/nlu-with-tensorrt-bert/ `