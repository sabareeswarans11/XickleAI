# XickleAI
XickleAI is a prototype fine-tuned language model designed to assist individuals with sickle cell disease (SCD) by providing reliable information, guidance, and support.

# Data Collection 
Since there is no publicly available dataset, especially for SCD, I need to scrape the data myself. The model's output mainly depends on the dataset it is trained on. I am using the Large Language Model Llama2:13b by meta-llama/Llama-2-13b-chat-hf to collect data in the form of questions and answers regarding SCD

* Data extraction includes web scraping, converting publications and books related to Sickle Cell Disease into a meaningful question-and-answer (Q&A) dataset using the Langchain concept
  * https://www.sc101.org/
  * https://www.cdc.gov/
  * Report. Institute for Clinical and Economic Review, July 13, 2023. https://icer.org/assessment/sickle-cell-disease-2023/
  * In Clinical Practice-Sickle Cell Disease in Clinical Practice Jo Howard Paul Telfer DOI 10.1007/978-1-4471-2473-3
  * dc.gov/ncbddd/sicklecell/facts.html
  * https://www.nhlbi.nih.gov/health/sickle-cell-disease 
  * https://my.clevelandclinic.org/health/diseases/12100-sickle-cell-disease
  * https://www.sicklecelldisease.org/
  * https://www.hhs.gov/ash/osm/innovationx/human-centered-design/sickle-cell-disease/index.html
  * Sickle Cell Disease and Hematopoietic Stem Cell Transplantation DOI 10.1007/978-3-319-62328-3
  * Newborn Screening for Sickle Cell Disease and other Haemoglobinopathies ISBN 978-3-03921-615-4 (PDF)


# Setting up and running LLM --xickle_datagen locally 
[Download-oolama- ThankYou](https://ollama.ai/download/Ollama-darwin.zip) -- Get up and running with large language models locally (MAC OS)
* Modelfile for creating a question and answer data generator assistant
*  ```ollama create xickle_datagen -f ./Modelfile``` and then
*  ```ollama run xickle_datagen``` and enter a topic
*  <img width="1000" alt="Screenshot 2023-10-09 at 12 45 35 AM" src="https://github.com/sabareeswarans11/XickleAI/assets/94094997/dc09693d-be6f-46c9-b0a4-74aeada94f34">
*  <img width="1000" alt="Screenshot 2023-10-09 at 12 42 34 AM" src="https://github.com/sabareeswarans11/XickleAI/assets/94094997/f7efc8e7-449f-413c-b89a-082bc5baaed0">

# FineTuned Model 
* I am currently working on the fine-tuning of Falcon-13b-instruct for XickleAI. This is just a starting point.
* I have fine-tuned a sample test model with Falcon-7 billion parameters. Although I used a relatively small dataset, its performance is enhanced due to its domain knowledge.
* GPT 3.5 vs XickleAI [scd_beta](https://huggingface.co/Sab11/scd_beta)
* <img width="950" alt="Screenshot 2023-10-09 at 1 00 57 AM" src="https://github.com/sabareeswarans11/XickleAI/assets/94094997/29920a21-1f5b-47bb-8ddc-ab7fe71804cc"> 
* <img width="950" alt="Screenshot 2023-10-09 at 12 15 36 AM" src="https://github.com/sabareeswarans11/XickleAI/assets/94094997/11f383fa-31b7-4e36-8737-9250631d7ac0">
* Note `This model is currently in the developing stage`

# Goal and Vision XickleAI
An AI assistant that empowers individuals affected by sickle cell disease with comprehensive support, from symptom tracking to research collaboration, enhancing their quality of life.
* Future goals
  * **Transition to vector embedding** and a vector database for improved efficiency, and fine-tuning with the latest large language model for ongoing enhancements
  * **Symptom Tracking**: Allow users to input and track their symptoms over time. This data can be useful for both individuals and their healthcare providers.
  * **Medication Reminders**: Set up medication reminders to help users stay on top of their treatment plans and offer information about common SCD medications and potential side effects.
  * **Research Updates**: Provide access to the latest research papers, clinical trials, and studies related to sickle cell disease.
  * **Appointment Scheduling**: Allow users to schedule appointments with their healthcare providers and send reminders.
 
# Acknowledgments
I would like to express my gratitude to the following **OpenSource-Community & Tools** to kickstart this project.
1. Hugging Face, especially [bitsandbytes integration](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#bitsandbytes-integration-for-fp4-mixed-precision-inference) 
2. [ollama](https://github.com/jmorganca/ollama/tree/main)  
3. [google colab](https://colab.google/) 
   





