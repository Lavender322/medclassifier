# Medical classifier

This is a medical classifier which takes either paragraphs or question
and classifies them in 7 labels - Overview or Epidemiology (depending on the model choice), 
Presentation, Diagnosis, Management, Medications, Follow up and Others.

To run the model, first install all dependencies from requirements.txt.

Then you have to download the weights folder from this URL: 
https://drive.google.com/drive/folders/1rulEfYDYLwsVAhgmCuV4UTkiYbye2B-v?usp=sharing 
and paste it into the home directory.

Then simply run from the home directory:
```
uvicorn api:app --reload
```
This will automatically instantiate the 127.0.0.1:8000, so you can simply send a POST request
to that address. If you want, you can select --host or --port to create a custom IP and port from
which the service will be run. 

After that, just experiment sending different curl requests to the service.
An example is shown below:
```
curl '127.0.0.1:8000/classify' --data '{"query": "How quickly do neurocognitive symptoms resolve in patients with long Covid?"}'
```
Note that I used /classify tag. This corresponds to the first model, which can output Overview separately. 
However if you use /classify_nosummary, the overview is no longer a possible output, only epidemiology, which is a
subsection of Overview, but does not also include Overview/Summary like previously.
