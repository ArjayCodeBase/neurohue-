pip install fastapi "uvicorn[standard]" sqlalchemy pydantic passlib[bcrypt] python-multipart Pillow email-validator python-dotenv

pip install --upgrade black pytest


uvicorn main:app --reload

python -m venv myenv

myenv\Scripts\activate

deactivate



echo ".venv/" >> .gitignore
echo ".env" >> .gitignore
echo "*.db" >> .gitignore
git add .
git commit -m "tester"
git push origin main

git pull origin main --rebase