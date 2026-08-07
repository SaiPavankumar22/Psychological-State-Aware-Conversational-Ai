# 🔐 Qdrant Secret Setup Guide

## Quick Setup

### Step 1: Create Modal Secret

```bash
modal secret create qdrant-credentials
```

### Step 2: Enter Environment Variables

When prompted, enter **THREE** environment variables (one per line):

```bash
QDRANT_URL=https://af09c824-3d37-4a40-a7ef-fc9bedda7f0b.us-east4-0.gcp.cloud.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your-actual-api-key-here
```

**Replace `your-actual-api-key-here` with your real Qdrant API key!**

---

## Getting Your Qdrant API Key

1. Go to [Qdrant Cloud Console](https://cloud.qdrant.io/)
2. Login to your account
3. Select your cluster: `af09c824-3d37-4a40-a7ef-fc9bedda7f0b`
4. Click **"API Keys"** in the left sidebar
5. Click **"Create New Key"** button
6. Copy the generated API key
7. Use it in the secret setup above

---

## Verify Secret

```bash
# List all secrets
modal secret list

# You should see:
# qdrant-credentials ✅
```

---

## Test Connection

After creating the secret, test the connection:

```bash
modal run realtime_conversational_ai.py::ConversationalAI.test_memory_pipeline
```

**Expected output:**
```
🔗 Connecting to Qdrant: https://af09c824-3d37-4a40-a7ef-fc9bedda7f0b.us-east4-0.gcp.cloud.qdrant.io:6333
🧠 Loading sentence-transformers model...
✅ Embedder loaded
📦 Creating collection: episodic_memory
✅ Created episodic_memory
📦 Creating collection: semantic_memory
✅ Created semantic_memory

🧪 Running memory pipeline tests...
✅ PASS: insert_episodic
✅ PASS: retrieve_episodic
...
✅ MEMORY SYSTEM FUNCTIONAL
```

---

## Troubleshooting

### Error: "QDRANT_API_KEY not found"

**Cause:** Secret not created or missing variables

**Fix:**
```bash
# Recreate with --force flag
modal secret create qdrant-credentials --force

# Enter all THREE variables again
```

### Error: "Connection refused" or "Timeout"

**Cause:** Wrong URL or PORT

**Fix:**
```bash
# Verify your cluster URL in Qdrant Cloud Console
# It should be: https://YOUR-CLUSTER-ID.REGION.gcp.cloud.qdrant.io

# Standard port is 6333
```

### Error: "Authentication failed"

**Cause:** Invalid API key

**Fix:**
1. Generate a new API key in Qdrant Cloud Console
2. Update the secret:
```bash
modal secret create qdrant-credentials --force
# Enter the NEW API key
```

---

## Connection Code Pattern

The system uses this pattern (as per your example):

```python
from qdrant_client import QdrantClient
import os

qdrant_client = QdrantClient(
    url=f"{os.environ['QDRANT_URL']}:{os.environ['QDRANT_PORT']}",
    api_key=os.environ["QDRANT_API_KEY"],
)
```

This is already implemented in `memory_store.py` (lines 77-81).

---

## Complete Example

Here's the full setup command:

```bash
# 1. Create secret
modal secret create qdrant-credentials

# 2. When prompted, paste:
QDRANT_URL=https://af09c824-3d37-4a40-a7ef-fc9bedda7f0b.us-east4-0.gcp.cloud.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=eyJhbGc... (your actual key)

# 3. Verify
modal secret list

# 4. Test
modal run realtime_conversational_ai.py::ConversationalAI.test_memory_pipeline

# 5. Deploy
modal deploy realtime_conversational_ai.py
```

---

## Notes

- **Port 6333** is the standard Qdrant HTTPS port
- The URL should include `https://` protocol
- API keys are sensitive - never commit them to git
- You can rotate keys anytime in Qdrant Cloud Console

---

✅ **Ready!** Once the secret is set up correctly, your memory system will connect automatically on container startup.
