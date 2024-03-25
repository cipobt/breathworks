import os
from utils import persona_description, breathworks_description, lda_keywords, topic_insights
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
        self.role = persona_description
        self.context = breathworks_description
        self.lda_keywords = lda_keywords
        self.topic_insights = topic_insights

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

    def identify_topics(self, query):
        """
        Identifies relevant topics based on query keywords.
        """
        query_keywords = query.lower().split()  # Simple split-based tokenization
        topic_relevance = {}
        # Directly iterate over the list of keywords, using enumerate to get the topic index (id)
        for topic_id, keywords_str in enumerate(self.lda_keywords['Keywords']):
            relevance_score = sum(keyword in query_keywords for keyword in keywords_str.split(", "))
            topic_relevance[topic_id] = relevance_score

        N = 3  # Define N, the number of top relevant topics to return

        relevant_topics = sorted(topic_relevance, key=topic_relevance.get, reverse=True)[:N]
        # Convert topic IDs (integers) to strings to match keys in self.topic_insights
        relevant_topic_strs = [str(topic) for topic in relevant_topics]
        return relevant_topic_strs

    def get_insights_for_topics(self, topics):
        """
        Fetches insights or summaries for the given list of topic IDs.
        This method assumes you have a predefined way of accessing insights
        for each topic, possibly precomputed.

        :param topics: List of topic IDs.
        :return: String containing combined insights for the topics.
        """
        insights = []
        for topic in topics:
            # Assuming `topic_insights` is a dict mapping topic IDs to insights
            # This could be replaced with any method you have to access insights
            topic_key = str(topic)
            if topic_key in self.topic_insights:
                insights.append(f"Topic {topic}: {self.topic_insights[topic_key]}")
            else:
                insights.append(f"Topic {topic}: No insights available.")
        return "\n".join(insights)

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

    def generate_response(self, query, max_tokens=100, use_lda_insights=True):
        """
        Generates a response from the model based on the hardcoded role and context, the provided query,
        and optionally insights from relevant LDA topics.

        :param query: The query to ask the model.
        :param max_tokens: The maximum number of tokens to generate.
        :param use_lda_insights: Whether to incorporate LDA topic insights into the response.
        :return: The generated response from the model.
        """
        if not self.adapter:
            print("No adapter created. Please create an adapter first.")
            return

        templated_query = f"### Role:\n{self.role}\n\n### Context:\n{self.context}\n\n### Query:\n{query}\n"

        # Incorporate LDA topic insights if requested
        if use_lda_insights:
            relevant_topics = self.identify_topics(query)
            insights = self.get_insights_for_topics(relevant_topics)
            templated_query += f"\n### Insights:\n{insights}\n"

        templated_query += "\n### Response:\n"

        try:
            response = self.adapter.complete(query=templated_query, max_generated_token_count=max_tokens)
            return response.generated_output
        except Exception as e:
            print(f"Error generating response: {e}")


if __name__ == "__main__":
    handler = ModelHandler()
    handler.list_models()
    handler.create_model_adapter(name="TestAdapter")

    # Generate and print the original response without using LDA topics
    original_response = handler.generate_response(
        query="What did you most like about Breathworks?",
        use_lda_insights=False  # Original answer without LDA topics
    )
    print(f"> Original Response (without LDA insights):\n{original_response}\n")

    # Generate and print the enhanced response using LDA topics
    enhanced_response = handler.generate_response(
        query="What did you most like about Breathworks?",
        use_lda_insights=True  # Enhanced answer with LDA topics
    )
    print(f"> Enhanced Response (with LDA insights):\n{enhanced_response}\n")

    handler.delete_adapter()
