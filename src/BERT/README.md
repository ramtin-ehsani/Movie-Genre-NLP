# BERT
Using BERT for doing the task of Conditional Natural Langauge Generation by fine-tuning pre-trained BERT on custom dataset. 

### Running the Server
```
1. Install the necessary requirements
2. Download the fine-tuned BERT model.
3. Run python3 app.py
4. Open index.html in the browser
```

### Tweaking the Parameters
1. You can tweak in __length__ of the text you want to generate by using the __slider__ shown in the demo image. The granularity of the slider is at __word__ level. Currently the limit is set to __maximum of 100 words__ at a time.
2. You can choose between __Random Hop__ and __Left to Right__ generation schemes. __Random Hop__ is usually seen to perform better than Left to Right.

### Source for sentence generation part (html+server):

https://github.com/prakhar21/Writing-with-BERT
