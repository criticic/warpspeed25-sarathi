# Agentic AI Assistant for WhatsApp

This Project was built at the [Warpspeed25 Hackathon](https://warpspeed2025.devfolio.co/) organized by [Lightspeed India](https://x.com/LightspeedIndia). It won the [AWS Track Prize for building a HealthTech Vernacular Health Record Navigator](https://warpspeed2025.devfolio.co/prizes?partner=Amazon+Web+Services). You can find the project details [on the Devfolio Project Page](https://warpspeed2025.devfolio.co/projects/sarathi-c7eb).

**Making sense of anything for India's 90%, in their language, right on WhatsApp.**

This project is a powerful, multilingual agentic assistant built to run on WhatsApp. It's designed to empower a vast majority of Indians, especially the elderly and those less comfortable with complex technology, by providing a simple, conversational way to understand complex information and access general knowledge.

The assistant leverages the powerful capabilities of **Google's Gemini** for reasoning and **Sarvam AI** for its excellent Indic language support (Speech-to-Text and Text-to-Speech), all orchestrated using **LangGraph**.

---

## 🚀 Key Features

* **Truly Multilingual & Voice-First:** Users can send voice notes or text in their native language, and the assistant will comprehend and respond in the same language and script, including voice replies.
* **Document & Image Analysis:** Understands complex information by simply sending a photo or a PDF.
  * **Legal Documents:** Explains dense legal papers in simple terms.
  * **Medical Prescriptions:** Identifies medicines and explains their purpose and dosage.
  * **Financial Statements:** Demystifies financial jargon.
* **General Knowledge Engine:** Can answer everyday questions about sports, entertainment, current events, and more, just like a search engine but in a conversational format.
* **Familiar & Accessible Interface:** By operating on WhatsApp, it eliminates the learning curve, making it instantly usable for hundreds of millions of people.

---

## 🛠️ Tech Stack

* **LLMs & AI Services:** Google Gemini, Sarvam AI (for Indic STT/TTS)
* **Backend:** Python
* **Frameworks:** LangGraph (for agent orchestration), FastAPI (for webhook server)
* **Platform:** Twilio API for WhatsApp
* **Server:** Uvicorn
* **Package Manager:** `uv`

---

## ⚙️ Getting Started

Follow these instructions to get a local copy up and running for development and testing.

### Prerequisites

* Python 3.11+ and `uv` installed.
* A [Twilio Account](https://www.twilio.com/try-twilio) with the WhatsApp Sandbox configured.
* API Keys for **Google Gemini** and **Sarvam AI**.
* [Ngrok](https://ngrok.com/download) or a similar tunneling service to expose your local server to the internet.

### Installation & Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/criticic/warpspeed25-sarathi.git
    cd warpspeed25-sarathi
    ```

2. **Create a `.env` file:**
    Create a file named `.env` in the root of the project and add your credentials. This keeps your secret keys out of the code.

    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
    SARVAM_API_KEY="YOUR_SARVAM_AI_API_KEY"
    TWILIO_ACCOUNT_SID="YOUR_TWILIO_ACCOUNT_SID"
    TWILIO_AUTH_TOKEN="YOUR_TWILIO_AUTH_TOKEN"
    ```

3. **Install dependencies:**
    Use `uv` to install all the required Python packages listed in your `pyproject.toml`.

    ```sh
    uv pip install .
    ```

4. **Expose your local server:**
    Run `ngrok` to create a public URL that tunnels to your local server on port 8000.

    ```sh
    ngrok http 8000
    ```

    Copy the `https://...` forwarding URL provided by ngrok.

5. **Configure the Twilio Webhook:**
    * Go to your [Twilio Console](https://console.twilio.com/).
    * Navigate to `Messaging -> Try it out -> Send a WhatsApp message`.
    * In the **Sandbox settings** tab, find the field "WHEN A MESSAGE COMES IN".
    * Paste your `ngrok` URL, appending the webhook endpoint to it. For example:
        `https://your-unique-ngrok-id.ngrok-free.app/api/v1/twilio/webhook`
    * Make sure the method is set to `HTTP POST`.
    * Click **Save**.

6. **Run the application server:**
    Now, start the Uvicorn server. The `--reload` flag will automatically restart the server when you make changes to the code.

    ```sh
    uv run uvicorn agent.webhook:app --reload
    ```

Your agent is now live!

---

## 📱 Usage

1. To start a conversation, send the sandbox join code (e.g., `join wide-angle`) from your WhatsApp to the Twilio number provided in your sandbox dashboard.
2. Once you're connected, you can:
    * Send a text message in any supported language.
    * Record and send a voice note.
    * Attach an image or a PDF document.
3. The assistant will process your input and reply in the same conversation.
