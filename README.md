# NLPExercise1
This is our implementation of a Skip-Gram model with negative sampling.

NB: We have a mistake in the test-format.sh. 
**Concerning the test script we have an issue running it. The condition where the script verifies if the similarity values are numbers is an error but the values are numbers**. So, I think it will be better to use your own `test_format.sh`
To generate the `tar.gz` skipGram file use this command :
```
  tar cvf skipGram.tar.gz skipGram.py
  ```
  
We used Docker as our algorithm didn’t work on Windows. Therefore, we created a container under Linux in order to execute the proper command lines and have the proper environment.
This is the `docker pull` command: 
```
  docker pull amirmahmoudi/nlp
  docker run -it <image-name> /bin/bash
  bash ./test_format.sh skipGram.tar.gz #error will be raised but the output is valid for the `results.txt`
  ```

The docker image is really heavy as all the environment is downloaded on the docker pull command, including all the packages, libraries and most importantly the billion words `.tar.gz` file.
**You should run the `skipGram.py` in this zip file using your own shell script, the docker is here to show the extra work we have done on this project.**

## Author

Chloé Daems<br/>
chloe.daems@student-cs.fr<br/>
Anne-Claire Laisney<br/>
anneclaire.laisney@student-cs.fr<br/>
Amir Mahmoudi <br/>
amir.mahmoudi@student-cs.fr<br/>
