# Deployment Guide - Production Hardened Emotion Psychology AI

## Quick Start for Modal Deployment

### 1. Pre-Deployment Checklist

```bash
# Verify no syntax errors
python -m py_compile realtime_conversational_ai.py memory_store.py

# Check all imports available
python -c "from realtime_conversational_ai import *"

# Verify Qdrant connection string is correct
echo $QDRANT_URL
echo $QDRANT_API_KEY
```

### 2. Set Up Modal Secrets

```bash
# Create emotion-env secret
modal secret create emotion-env \
  OPENAI_API_KEY="your-key-here" \
  AZURE_SPEECH_KEY="your-key-here" \
  AZURE_SPEECH_REGION="your-region"

# Create qdrant-credentials secret
modal secret create qdrant-credentials \
  QDRANT_URL="http://your-qdrant-host:6333" \
  QDRANT_API_KEY="your-api-key"
```

### 3. Deploy to Modal

```bash
# Deploy the application
modal deploy realtime_conversational_ai.py

# Expected output:
# ✓ Created new app 'conversational-ai'
# ✓ Image built and pushed
# ✓ Deployed web endpoint
# ✓ Deployed WebSocket endpoint

# Get deployment URL
modal run realtime_conversational_ai.py
```

### 4. Verify Deployment

```bash
# Test health endpoints
curl https://your-app.modal.run/health
# Expected: {"status": "healthy"}

curl https://your-app.modal.run/readiness
# Expected: {"status": "ready", "memory": "available"}

curl https://your-app.modal.run/liveness
# Expected: {"status": "alive"}

# Open web interface
open https://your-app.modal.run/
```

---

## Environment Configuration

### Required Secrets

#### emotion-env
```
OPENAI_API_KEY=sk-...              # OpenAI API key for LLM
AZURE_SPEECH_KEY=...               # Azure Cognitive Services key
AZURE_SPEECH_REGION=eastus         # Azure region (e.g., eastus, westus2)
```

#### qdrant-credentials
```
QDRANT_URL=http://localhost:6333   # Qdrant vector DB URL
QDRANT_API_KEY=...                 # Qdrant API key (if auth enabled)
```

### Optional Tuning Parameters

In `realtime_conversational_ai.py`:
- Line 777: `rate_limit_window = 2.0` - Adjust rate limit (seconds per request)
- In `memory_orchestrator.py`: Adjust importance score thresholds
- In `emotional_trends.py`: Adjust EMA smoothing factor (0.4)

---

## Cold Start Optimization

### Build Time Breakdown (After Optimization)

```
Layer 1 (Base OS):        ~5s  (cached in subsequent builds)
Layer 2 (Core deps):      ~8s  (cached unless requirements.txt changes)
Layer 3 (ML models):      ~15s (cached unless model imports change)
Layer 4 (API deps):       ~8s  (cached unless imports change)
Layer 5 (Local files):    ~5s  (ALWAYS rebuilds - fastest layer)
═════════════════════════════════════
Total (first build):      ~45s
Total (code-only change): ~8-10s ✅ (only Layer 5 rebuilds)
```

### Tips for Faster Cold Starts

1. **Keep code changes frequent, dependencies rare**
   - Avoid adding new imports after model initialization
   - New dependencies go in Layer 2/3/4, invalidating cache

2. **Local development without Modal**
   ```bash
   # Test locally first
   python -c "from realtime_conversational_ai import ConversationalAI; ca = ConversationalAI()"
   ```

3. **Use Modal's run_app for iteration**
   ```bash
   modal run realtime_conversational_ai.py --detach
   ```

4. **Monitor cold start metrics**
   ```bash
   modal logs --level INFO realtime_conversational_ai
   ```

---

## Monitoring & Alerting

### Key Metrics to Track

```
✓ Latency: P50, P95, P99 for each component (ASR, Emotion, LLM, TTS)
✓ Errors: Count by type (memory, LLM, TTS, validation)
✓ Availability: /readiness endpoint latency and success rate
✓ Fallback Rate: Percentage of requests using degraded paths
✓ Memory Stats: Episodic vs semantic memory counts
```

### Log Patterns to Alert On

**CRITICAL (Page immediately):**
```
❌ Qdrant connection failed (retry exhausted after 30s)
❌ All LLM attempts failed (model overloaded?)
❌ WebSocket handler exceptions
```

**HIGH (Alert within 5 min):**
```
⚠️ Rate limit exceeded (possible abuse)
⚠️ Memory storage failed (repeated failures)
⚠️ TTS synthesis failed (audio service down)
```

**MEDIUM (Review in daily reports):**
```
⚠️ Empty transcript (user input issue or ASR failure?)
⚠️ Invalid LLM response (quality issue)
⚠️ Memory fallback mode active (Qdrant slow?)
```

---

## Troubleshooting

### Issue: "Qdrant unavailable" in logs but system still working

**Status:** ✅ Expected behavior - fallback mode active
**Action:** Check Qdrant service connectivity
```bash
curl http://your-qdrant-url:6333/health
# Should return 200 with health status
```

**Fix:**
- Verify QDRANT_URL and QDRANT_API_KEY secrets
- Check network connectivity from Modal to Qdrant
- Restart Qdrant service if needed

### Issue: Conversation turns taking >10s

**Investigate:**
```
- Check logs for "Processing - Session" to turn completion time
- Check Azure TTS latency (should be <2s)
- Check LLM latency (should be <3s)
- Check Qdrant query latency (should be <1s)
```

**Common Causes:**
- LLM API overloaded (add backoff/exponential retries)
- Large audio file (WAV uncompressed - consider MP3)
- Qdrant memory retrieval slow (check database size/indexing)

### Issue: "Rate limit exceeded" appears legitimate users

**Solution:** Adjust rate limit window in code
```python
# Increase from 2.0s to 5.0s
rate_limit_window = 5.0
```

Then redeploy:
```bash
modal deploy realtime_conversational_ai.py
```

### Issue: WebSocket connection fails immediately

**Possible causes:**
- Invalid session_id format
- Missing audio data
- JSON parsing error on client side

**Debug:**
```bash
# Check WebSocket handshake
modal logs -q realtime_conversational_ai | grep WebSocket

# Check exact error
modal logs realtime_conversational_ai | grep "1008\|1011"
```

### Issue: Memory system not working (always "unavailable")

**Check health:**
```bash
curl https://your-app.modal.run/readiness
# If {"memory": "unavailable"}, Qdrant is down
```

**Verify connection:**
```bash
# From Modal pod
modal exec realtime_conversational_ai -- \
  python -c "from memory_store import MemoryStore; \
  m = MemoryStore(); print(m.is_available())"
```

---

## Performance Tuning

### If Conversations Too Slow

1. **Reduce memory retrieval scope**
   ```python
   # In realtime_conversational_ai.py line 376
   max_memories=6  # Reduce to 3-4
   ```

2. **Disable semantic extraction**
   ```python
   # In memory_orchestrator.py
   if self.memory_orchestrator.should_extract_semantic_memory(...):
       # Comment out extraction during high load
   ```

3. **Increase TTS timeout**
   - If TTS frequently failing, check audio encoding

### If Memory Usage High

1. **Reduce session retention**
   ```python
   # In realtime_conversational_ai.py
   # Increase cleanup frequency for old sessions
   ```

2. **Limit episodic memory retention**
   ```python
   # In memory_orchestrator.py
   max_episodic_per_user = 1000
   ```

### If Cold Start Still Slow

1. **Separate heavy models into separate endpoints**
   ```python
   @modal.function()  # Lightweight function
   async def transcribe(audio):
       # Light preprocessing
       pass
   
   @modal.function(gpu="A10G")  # Heavy function
   async def detect_emotion(audio):
       # GPU-required analysis
       pass
   ```

2. **Use Modal's warm pool**
   ```python
   @app.cls(
       image=image,
       gpu="A10G",
       keep_warm=2  # Keep 2 instances warm
   )
   ```

---

## Rollback Procedure

If deployment has issues:

```bash
# View deployment history
modal objects list --apps

# Redeploy previous version (if git tag exists)
git checkout previous-tag
modal deploy realtime_conversational_ai.py

# Or immediately revert to known-good version
git revert HEAD
modal deploy realtime_conversational_ai.py

# Verify rollback succeeded
curl https://your-app.modal.run/health
```

---

## Success Criteria

After deployment, verify:

- [ ] All health check endpoints respond within 500ms
- [ ] WebSocket connections establish within 2s
- [ ] Conversation turns complete within 10s (average)
- [ ] No "❌ Error" logs in first 5 minutes
- [ ] Memory system available ("⚠️ Memory retrieval failed" ok, but should be rare)
- [ ] Rate limiting working (test with rapid requests)
- [ ] Qdrant connection successful (check logs)
- [ ] All required secrets configured
- [ ] Cold start time <25s (code-only builds)

---

## Support & Debugging

**For detailed debugging:**
```bash
# Stream logs in real-time
modal logs -f realtime_conversational_ai --level DEBUG

# Run test function
modal run realtime_conversational_ai.ConversationalAI.test_memory_pipeline

# Execute shell command in pod
modal exec realtime_conversational_ai -- bash
```

**Contact/Resources:**
- Modal Docs: https://modal.com/docs
- Qdrant Docs: https://qdrant.tech/documentation
- Azure Speech: https://learn.microsoft.com/azure/cognitive-services

---

**Deployment Status:** ✅ Ready for Production  
**Last Updated:** After production hardening session  
**Next Steps:** Deploy to staging, load test, then production
