# Environment Setup for ChatKit

To use the ChatKit AI Assistant in the Social Media Dashboard, you need to set up your OpenAI API key.

## Quick Setup

1. Create a `.env` file in the project root directory (same level as `main.py`)

2. Add the following line to your `.env` file:
   ```
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

3. Replace `your_actual_openai_api_key_here` with your actual OpenAI API key from https://platform.openai.com/api-keys

4. Save the file and restart your Streamlit app

## Example .env file

```
# OpenAI API Configuration
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: ChatKit Server URL (if using custom backend)
# CHATKIT_SERVER_URL=https://your-chatkit-server.com
```

## Notes

- The `.env` file is automatically ignored by git (it's in `.gitignore`) so your API key won't be committed to version control
- You can also set the `OPENAI_API_KEY` as a system environment variable if you prefer
- If the API key is not found in `.env`, you'll be prompted to enter it in the dashboard configuration section

