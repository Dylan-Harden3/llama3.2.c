## llama3.2.c - clone of Karpathy's llama2.c but updated to work with Llama 3.2 family models.

You can see [the original repo](https://github.com/karpathy/llama2.c) for a fullstack train+inference solution for llama2. I only updated the inference code [run.c](run.c) to use Llama 3.2 so if you want to train you will need to update the repo yourself.

<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

Have you ever wanted to inference a [Llama 3.2](https://www.llama.com/) model in pure C? No? Well, now you can!

Inference Llama 3.2 1B/3B with one simple 700-line C file ([run.c](run.c)). You might think that you need many billion parameter LLMs to do anything useful, but in fact very small LLMs can have surprisingly strong performance if you make the domain narrow enough (ref: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) paper). This repo is an inference solution for Llama3.2 family LLMs, with focus on minimalism and simplicity.

## feel the magic
First, navigate to the folder where you keep your projects and clone this repository to this folder:

```bash
git clone https://github.com/karpathy/llama2.c.git
```

Then, open the repository folder:

```bash
cd llama2.c
```

The only prerequisites are gcc (or clang) and [PCRE](https://www.pcre.org/) - used for regex splitting in the new tokenizer (from scratch regex is outside the scope of this project).
I installed PCRE with:
```bash
sudo apt install libpcre3 libpcre3-dev
```

Now, let's just run a Llama 3.2 model in C. You need a model checkpoint. Use [export.py](export.py) to load the models from huggingface. You need to apply for access and login with [huggingface cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

```bash
python3 export.py Llama-3.2-1B.bin --hf meta-llama/Llama-3.2-1B
```

Compile and run the C code:

```bash
make run
./run Llama-3.2-1B.bin
```

You'll see the text stream a sample. On my machine this runs at ~9 tokens/s. See [performance](#performance) or the Makefile for compile flags that can significantly speed this up. We can also try a bit bigger 3B parameter model:

```bash
python3 export.py Llama-3.2-3B.bin --hf meta-llama/Llama-3.2-1B
./run Llama-3.2-3B.bin
```

You can also prompt the model with a prefix or a number of additional command line arguments, e.g. to sample at temperature 0.8 for 256 steps and with a prompt:

```bash
./run Llama-3.2-1B.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
```

Quick note on sampling, the recommendation for ~best results is to sample with `-t 1.0 -p 0.9`, i.e. temperature 1.0 (default) but also top-p sampling at 0.9 (default). Intuitively, top-p ensures that tokens with tiny probabilities do not get sampled, so we can't get "unlucky" during sampling, and we are less likely to go "off the rails" afterwards. More generally, to control the diversity of samples use either the temperature (i.e. vary `-t` between 0 and 1 and keep top-p off with `-p 0`) or the top-p value (i.e. vary `-p` between 0 and 1 and keep `-t 1`), but not both. Nice explainers on LLM sampling strategies include [this](https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/), [this](https://docs.cohere.com/docs/controlling-generation-with-top-k-top-p) or [this](https://huggingface.co/blog/how-to-generate).

You can also chat with the Llama 3.2 Instuction Tuned Chat models. Export the chat model exactly as above:

```bash
python export.py Llama-3.2-1B-Instruct.bin --hf meta-llama/Llama-3.2-1B-Instruct
```

Then chat with it by specifying the chat mode using the `-m` flag, e.g.:

```bash
./run Llama-3.2-1B-Instruct.bin -m chat
```

## int8 quantization

The (default) script [run.c](run.c), above, uses a float32 forward pass, where the entire calculation of the forward pass is kept in fp32. This is very easy to understand as far as reference code goes, but it has the following downsides: the model checkpoint files are very large (it takes 4 bytes per every individual weight), and the forward pass is relatively slow. The (very) common inference optimization employed in practice is to quantize the model parameters to lower precision, giving up a little bit of correctness in return for smaller checkpoint sizes and faster forward passes (as most of the inference uses integer arithmetic). Empirically, LLMs can tolerate precisions as low as 4-bit (or even lower), but we use int8 here because it is a "safe" setting that gets us the benefits but doesn't sacrifice too much of the model accuracy. Only the weights that participate in matmuls are quantized. All the other parameters (e.g. especially the scale and bias in RMSNorm) are kept in float32, because these layers are very sensitive. Now, if all you're after is reduction in checkpoint sizes, you could quantize the weights, save the checkpoint, and then dequantize them in run.c, and do float32 inference as normal and call it a day. This is totally fine. But here, we go one step further (as is standard practice) and additionally quantize the activations in the forward pass. This requires us to dynamically quantize and dequantize between float32 and int8 at runtime, which adds overhead. But the benefit is that now the majority of the calculations (the matmuls especially!) are using pure integer arithmetic, where both weights and activations enter as int8. This is where the speedups can fundamentally come from. The version we use is the "Q8_0" quantization (llama.cpp terminology), where the 0 means that the weight quantization is symmetric around 0, quantizing to the range [-127, 127].

The quantized forward pass is implemented in [runq.c](runq.c). To use it, we have to export the model in the quantized format. For example, the float32 version of Llama 3.2 1B was exported as:

```
python export.py Llama-3.2-1B.bin --hf meta-llama/Llama-3.2-1B
```

This creates a 4.7GB file, because each one of 1B parameters is 4 bytes (fp32). To export it quantized, we instead use version 2 export:

```
python export.py Llama-3.2-1B-q8_0.bin --version 2 --hf meta-llama/Llama-3.2-1B
```

This runs for a few minutes, but now creates only a 1.3GB file. Now let's inference them. I like to use OMP here because these are big models, so e.g. on my machine:

```
make runomp
OMP_NUM_THREADS=6 ./run Llama-3.2-1B.bin -n 40
OMP_NUM_THREADS=6 ./runq Llama-3.2-1B-q8_0.bin -n 40
```

This runs 40 steps just to get a timing. The float32 version for me runs at 9 tok/s, and the int8 version at 26 tok/s. So we achieved a 3X speedup while reducing the checkpoint size by 4X. However, the forward pass is quantized to int8, and therefore silently very slightly lower quality.

## huggingface models

We can load any huggingface models that use the Llama 3.2 architecture. See the script [export.py](export.py) and the `--hf` flag to export the model .bin file.

## performance

There are many ways to potentially speed up this code depending on your system. Have a look at the [Makefile](Makefile), which contains a lot of notes. The `make run` command currently uses the `-O3` optimization by default, i.e.:

```bash
gcc -O3 -o run run.c -lm -lpcre
```

-O3 includes optimizations that are expensive in terms of compile time and memory usage. Including vectorization, loop unrolling, and predicting branches.

To get a much better performance, try to compile with `make runfast`. This turns on the `-Ofast` flag, which includes additional optimizations that may break compliance with the C/IEEE specifications, in addition to `-O3`. See [the GCC docs](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) for more information.

Try `-march=native` to compile the program to use the architecture of the machine you're compiling on rather than a more generic CPU. This may enable additional optimizations and hardware-specific tuning such as improved vector instructions/width.

The fastest throughput I saw so far on my machine so far is with `make runomp`.

You can also experiment with replacing `gcc` with `clang`.

If compiling with gcc, try experimenting with `-funroll-all-loops`, see PR [#183](https://github.com/karpathy/llama2.c/pull/183)

**OpenMP**. Big improvements can also be achieved by compiling with OpenMP, which "activates" the `#pragma omp parallel for` inside the matmul and attention, allowing the work in the loops to be split up over multiple processors.
You'll need to install the OpenMP library and the clang compiler first (e.g. `apt install clang libomp-dev` on ubuntu). Then you can compile with `make runomp`, which does:

```bash
clang -Ofast -fopenmp -march=native run.c  -lm -lpcre  -o run
```

When you run inference make sure to use OpenMP flags to set the number of threads, e.g.:

```bash
OMP_NUM_THREADS=4 ./run out/model.bin
```

Depending on your system resources you may want to tweak these hyperparameters and use more threads. But more is not always better, usually this is a bit U shaped. In particular, if your CPU has SMT (multithreading), try setting the number of threads to the number of physical cores rather than logical cores. The performance difference can be large due to cache thrashing and communication overhead. The PyTorch documentation [CPU specific optimizations
](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#cpu-specific-optimizations) has some good information that applies here too.

## License

MIT
