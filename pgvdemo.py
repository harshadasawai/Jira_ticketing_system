import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import requests
from requests.auth import HTTPBasicAuth
from transformers import AutoTokenizer, AutoModel
import torch
import json

import psycopg2
from psycopg2.extras import execute_values

load_dotenv()

# Database connection settings
DB_NAME = "test_db"
DB_USER = "admin_user"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = "5432"

# Connect to PostgreSQL and ensure pgvector extension is installed
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS jira_embeddings (
        ticket_id TEXT PRIMARY KEY,
        embedding vector(768),
        metadata JSONB
    );
    """
)
conn.commit()

# Set your API key and configure the model
api_key = "AIzaSyBKx3HDaOcDwcU6SB7plkiXDHU7_r8p6Nc"
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 0.9,
    "top_p": 0.5,  
    "top_k": 5,     
    "max_output_tokens": 1000,  
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

chat = None

# Jira credentials
JIRA_API_TOKEN = "ATATT3xFfGF0C-1pgO_Txi3xnprWHyMIRW9XP1Z5Cq8IgWgM8KL973Kds7aNidcn3CQ2nkYOLfpIy8zroG6xKjaOl9i8FURDRNiWfm0-tirpyL2tfkeuPwP_C_r9PR7L6AjJ0_a5BGAMmvRwmywEeAR57q6iFonAH4R9JQXo0DuY9FK54oSrs7Y=B14D3A92"
JIRA_EMAIL = "ajinkya.matre23@vit.edu"
JIRA_BASE_URL = "https://vit-team-azyltavp.atlassian.net"
JIRA_SEARCH_ENDPOINT = f"{JIRA_BASE_URL}/rest/api/3/search"

auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}


# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Fetch Jira tickets (done tickets)
def fetch_jira_tickets(status='done'):
    query = {'jql': f'status = "{status}"'}
    response = requests.get(JIRA_SEARCH_ENDPOINT, headers=headers, auth=auth, params=query)
    if response.status_code == 200:
        tickets = response.json()
        return tickets['issues']
    else:
        print(f"Failed to fetch tickets: {response.status_code}")
        return []

# Fetch comments for a specific Jira ticket
def fetch_ticket_comments(ticket_id):
    comments_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{ticket_id}/comment"
    response = requests.get(comments_url, headers=headers, auth=auth)
    if response.status_code == 200:
        comments = response.json().get('comments', [])
        return [comment['body']['content'][0]['content'][0].get('text', '') for comment in comments if 'body' in comment]
    else:
        print(f"Failed to fetch comments for ticket {ticket_id}: {response.status_code}")
        return []

# Get embeddings for text
def get_embeddings(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

# Function to generate a prompt using Jira ticket data
def generate_ticket_prompt(tickets):
    ticket_prompt = ""
    
    for ticket in tickets:
        ticket_id = ticket['id']
        issue_title = ticket['fields']['summary']
        description_obj = ticket['fields']['description']
        
        # Extract the plain text description
        description = get_ticket_description(description_obj)
        
        # Fetch comments (solutions) for the ticket
        comments = fetch_ticket_comments(ticket_id)
        
        # Join all comments into a single string
        all_comments_text = " ".join(comments) if comments else "No comments available"
        
        # Combine description and comments to form a complete solution
        complete_text = (
            f"Ticket ID: {ticket_id}\n"
            f"Title: {issue_title}\n"
            f"Description: {description}\n"
            f"Solution (Comments): {all_comments_text}\n"
        )
        ticket_prompt += complete_text + "\n\n"  # Separate tickets by newlines
    
    return ticket_prompt


# Function to extract description from a Jira ticket
def get_ticket_description(description_obj):
    if isinstance(description_obj, dict) and 'content' in description_obj:
        paragraphs = description_obj['content']
        description_text = []
        for paragraph in paragraphs:
            if 'content' in paragraph:
                for content in paragraph['content']:
                    if 'text' in content:
                        description_text.append(content['text'])
        return " ".join(description_text)
    return "No description available"



# Store ticket embeddings in PostgreSQL
def store_ticket_in_postgres(ticket):
    ticket_id = ticket['id']
    issue_title = ticket['fields']['summary']
    description_obj = ticket['fields']['description']
    description = get_ticket_description(description_obj)
    priority = ticket['fields']['priority']['name'] if 'priority' in ticket['fields'] else "Unknown"
    status = ticket['fields']['status']['name'] if 'status' in ticket['fields'] else "Unknown"

    # Fetch comments (solutions) for the ticket
    comments = fetch_ticket_comments(ticket_id)
    all_comments_text = " ".join(comments) if comments else "No comments available"

    # Create the full text representation
    full_text = f"{issue_title} {description} {all_comments_text}"

    # Generate the embedding
    embedding = get_embeddings(full_text)

    # Prepare metadata
    metadata = {
        "description": description,
        "priority": priority,
        "status": status,
        "summary": issue_title
    }

        # Insert data into PostgreSQL
    cur.execute(
        """
        INSERT INTO jira_embeddings (ticket_id, embedding, metadata) 
        VALUES (%s, %s, %s)
        ON CONFLICT (ticket_id) DO UPDATE 
        SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata;
        """,
        (ticket_id, embedding.tolist(), json.dumps(metadata))
    )
    conn.commit()



# Chatbot function to solve Jira tickets
def resolve_jira_tickets_chatbot(conversation_history):
    tickets = fetch_jira_tickets(status="Done")
    prompt_data = generate_ticket_prompt(tickets)

    # Configure the chatbot with the Jira data
    model = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config, safety_settings=safety_settings)
    global chat
    
    if not chat:
        chat = model.start_chat()
    
    # Modify the prompt for ticket-related problem-solving
    prompt = f"Act as a Jira issue-solving assistant Having name Jira ticket solver. Here are some Jira tickets and their solutions: \n\n{prompt_data}\n\nNow, based on the above information and Using the above solutions, assist me in solving new Jira tickets or answering queries about existing ones.And Here's the conversation history (if histroy is not present yet means start of bot so print intro message) while giving answer of any query give priority for the solution of exitsting simiar tickets and then frame the answer(but don't show the similar detect just use its solution and mention the Ticket Number in the response of the simalar ticket and if dont have any simalar ticket dont mention the ticket no and give response):"
    

    for entry in conversation_history:
        prompt += f"{entry['sender']}: {entry['message']}\n"
    
    prompt += "\nNow respond to the latest query based on the above history."

    # Start the chatbot loop
    response = chat.send_message(prompt)
    

    return response.text

def text_summary(text, isNew=False, conversation_history=[]):
    """
    Summarize the given text or return it based on the 'isNew' flag.
    """
    if isNew:
        # For new conversations, return the first 100 characters as a summary
        return text[:100] + "..."
    else:
        # Add the current message to the conversation history and return chatbot's response
        conversation_history.append({'sender': 'user', 'message': text})
        return resolve_jira_tickets_chatbot(conversation_history)

def main():
    # Fetch Jira tickets and store embeddings in Pinecone
    tickets = fetch_jira_tickets(status="done")
    for ticket in tickets:
        store_ticket_in_postgres(ticket)
    
    # Run the chatbot for Jira ticket solving
    resolve_jira_tickets_chatbot([])

if __name__ == '__main__':
    main()
