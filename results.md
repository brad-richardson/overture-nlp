## Vandalism detection
Testing done with RTX4070 GPU

Model           |  n=20 accuracy | n=20 precision | n=20 recall | n=20 time (sec) | n=2000 accuracy | n=2000 time (min)
-----------------------------------------------------------------------------------------------------------------------
llama3.2:1b     |  62%           | 61%            | 100%        | 13              | 49.3%           | 20
llama3.2:3b     |  76%           | 89%            | 73%         | 15              | 79.8%           | 23
phi4:14b        |  71%           | 100%           | 55%         | 66              | 68.5%           | 100
deepseek-r1:14b |  62%           | 100%           | 30%         | 31              | 71.1%           | 42
Unable to test llama3.3:70b, deepseek-r1:32b, and mixtral:8x7b due to memory limitations

## Language validation
Testing done on M1 GPU
TODO - precision/recall

Model           |  n=20 accuracy | n=20 precision | n=20 recall | n=20 time (sec)
---------------------------------------------------------------------------------
llama3.2:1b     |  26.3%         | ??             | ??          | 14
llama3.2:3b     |  42.1%         | ??             | ??          | 31
phi4:14b        |  47.4%         | ??             | ??          | 220
deepseek-r1:14b |  31.6%         | ??             | ??          | 114
