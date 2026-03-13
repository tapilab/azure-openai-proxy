Chat log: https://chatgpt.com/c/696ec941-ced0-832f-8372-3ffaa8af8d86

To publish to Azure:

`func azure functionapp publish shared-openai-function`


To add a new model:
- Deploy a new model at https://ai.azure.com/ under this app
- Add a new entry to the AZURE_OPENAI_MODEL_MAP environment variable. 
	+ See portal.azure.com -> shared-openai-function -> Settings -> Environmental Variables