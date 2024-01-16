# emotion-detection

To download a Kaggle dataset using the Kaggle CLI, you can use the `kaggle datasets download` command. In your case, the dataset is hosted at `jonathanoheix/face-expression-recognition-dataset`. Here are the steps to download the dataset:

1. **Ensure you have the Kaggle CLI installed:**
   If you haven't installed the Kaggle CLI yet, you can do so using the following command:
   ```bash
   pip install kaggle
   ```

2. **Set up your Kaggle API credentials:**
   Make sure you have the `kaggle.json` file in the `~/.kaggle/` directory. If it's not there, move it or download it using the steps mentioned earlier.

3. **Download the dataset:**
   Use the `kaggle datasets download` command to download the dataset. Open your terminal and run the following command:
   ```bash
   kaggle datasets download -d jonathanoheix/face-expression-recognition-dataset
   ```

   This command will download the dataset as a zip file to your current directory.

4. **Extract the dataset:**
   After downloading, you'll likely want to extract the contents of the zip file. You can use the `unzip` command for this:
   ```bash
   unzip face-expression-recognition-dataset.zip
   ```

   This will extract the contents into the current directory.

Now you should have the dataset downloaded and extracted in your working directory. Adjust the paths or move the files as needed for your specific use case.