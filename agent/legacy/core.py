import os
from gradientai import Gradient
from dotenv import load_dotenv

load_dotenv()

GRADIENT_ACCESS_TOKEN = os.getenv('GRADIENT_ACCESS_TOKEN')
GRADIENT_WORKSPACE_ID = os.getenv('GRADIENT_WORKSPACE_ID')

class ModelHandler:
    def __init__(self):
        """
        Initialises the ModelHandler.
        """
        self.gradient = Gradient()
        self.base_model_slug = "nous-hermes2"
        self.adapter = None
        self.role = "You are a female, around 35 years old, living in the UK. You have enrolled in a course to improve mental health, most likely you were having some pain or going through a tough time."
        self.context = "Breathworks is a mentalhealth Charity based in the UK. It provides mentalhealth courses and would like to understand better your feeling and motivation to take future courses. The marketing team wants to understand more about their audience. "

    def list_models(self):
        """
        Lists all models available in the Gradient instance.
        """
        try:
            models = self.gradient.list_models(only_base=False)
            for model in models:
                if hasattr(model, "name"):
                    print(f"{model.name}: {model.id}")
        except Exception as e:
            print(f"Error listing models: {e}")

    def create_model_adapter(self, name):
        """
        Creates a model adapter for the base model.
        """
        try:
            base = self.gradient.get_base_model(base_model_slug=self.base_model_slug)
            self.adapter = base.create_model_adapter(name=name)
        except Exception as e:
            print(f"Error creating model adapter: {e}")

    def delete_adapter(self):
        """
        Deletes the model adapter if it exists.
        """
        if self.adapter:
            try:
                self.adapter.delete()
                print("Adapter deleted successfully.")
            except Exception as e:
                print(f"Error deleting adapter: {e}")


    def generate_response(self, query, max_tokens=100):
            """
            Generates a response from the model based on the hardcoded role and context, and the provided query.

            :param query: The query to ask the model.
            :param max_tokens: The maximum number of tokens to generate.
            :return: The generated response from the model.
            """
            if not self.adapter:
                print("No adapter created. Please create an adapter first.")
                return

            # Using the hardcoded role and context in the templated query
            templated_query = f"### Role:\n{self.role}\n\n### Context:\n{self.context}\n\n### Query:\n{query}\n\n### Response:\n"
            try:
                response = self.adapter.complete(query=templated_query, max_generated_token_count=max_tokens)
                return response.generated_output
            except Exception as e:
                print(f"Error generating response: {e}")


if __name__ == "__main__":
    handler = ModelHandler()
    handler.list_models()
    handler.create_model_adapter(name="TestAdapter")
    response = handler.generate_response(
        query="What would make you enrol in another course at Breathworks?"
    )
    print(f"> Response:\n{response}\n")
    handler.delete_adapter()
