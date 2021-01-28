echo "1. Installing Requirements"
echo "--------------------------"
pip install -r requirements.txt

echo "2. Downloading Spacy en_core_web_sm model"
echo "------------------------------------------"
python -m spacy download en_core_web_sm

echo "3. Downloading Processed Resources"
echo "----------------------------------"
curl -O https://ritual.uh.edu/wp-content/uploads/projects/res_emnlp2020_sk.tar.gz

echo "4. Unpacking Resources"
echo "-----------------------"
tar -xf res_emnlp2020_sk.tar.gz