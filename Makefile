install_requirements:
	@pip install -r requirements.txt

install:
	@pip install . -U

run_api:
	uvicorn sign_language.api.fast:app --reload
