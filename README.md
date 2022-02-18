# NLPExercise1
This is our implementation of a Skip-Gram model with negative sampling.

NB: We have a mistake in the test-format.sh. 
**Concerning the test script we have an issue running it. The condition where the script verifies if the similarity values are numbers is an error but the values are numbers**. So, we first run the script to download the `train.txt` and `simlex.csv` and we then run each python command by itself.

If you can take this into account or use your own `test_format.sh`. 
Here are the python command to run in the shell :
  ```
  python skipGram.py --model mymodel.model --text ./train.txt
  python skipGram.py --test --model mymodel.model --text ./simlex.csv > ./results.txt
  ```
  
The easiest method is to **test the command lines directly in the terminal**, which is what we did to test the billion words vocabulary. 

In our final repository version, you will find a `dockerfile`. We used it as our algorithm didn’t work on Windows. Therefore, we created a container under Linux in order to execute the proper command lines and have the proper environment.
This is the `docker pull` command: 
```
  docker pull amirmahmoudi/nlpexercise1
  docker run -it <image-name> /bin/bash
  bash ./test_format.sh skipGram.tar.gz #error will be raised as mentionned before but run the 2 next lines please
  python skipGram.py --model mymodel.model --text ./train.txt
  python skipGram.py --test --model mymodel.model --text ./simlex.csv > ./results.txt
  ```

The docker image is really heavy as all the environment is downloaded on the dockerpullcommand, including all the packages, libraries and most importantly the billion words `.tar.gz` file.

## Author

Chloé Daems<br/>
chloe.daems@student-cs.fr<br/>
Anne-Claire Laisney<br/>
anneclaire.laisney@student-cs.fr<br/>
Amir Mahmoudi <br/>
amir.mahmoudi@student-cs.fr<br/>
