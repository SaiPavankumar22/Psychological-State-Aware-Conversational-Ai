# Testing & Validation Guide

## Quick Validation (5 minutes)

```bash
# 1. Check syntax
python -m py_compile realtime_conversational_ai.py memory_store.py

# 2. Import test
python -c "from realtime_conversational_ai import ConversationalAI; print('✅ Imports OK')"

# 3. Check logging setup
python -c "import logging; logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__); logger.info('✅ Logging OK')"
```

**Expected:** All checks pass ✅

---

## Unit Tests (Needs Implementation)

Create `test_production_hardening.py`:

```python
import asyncio
import pytest
from memory_store import MemoryStore
from realtime_conversational_ai import ConversationalAI

# Test 1: Qdrant Fallback Mode
def test_memory_fallback_mode():
    """Test that memory system gracefully handles Qdrant unavailability"""
    store = MemoryStore(qdrant_url="http://localhost:9999")  # Wrong URL
    
    # Should not crash, should enter fallback mode
    assert store._fallback_mode == True
    assert store.is_available() == False
    
    # Operations should return None/empty instead of crashing
    result = store.retrieve_episodic_memories("user1", "query", limit=5)
    assert result == []  # Empty list, not exception

# Test 2: Rate Limiting
def test_rate_limiting():
    """Test that rate limiting prevents rapid requests"""
    from realtime_conversational_ai import check_rate_limit
    
    session_id = "test_session"
    
    # First request should pass
    assert check_rate_limit(session_id) == True
    
    # Immediate second request should fail
    assert check_rate_limit(session_id) == False
    
    # After 2.1 seconds should pass
    time.sleep(2.1)
    assert check_rate_limit(session_id) == True

# Test 3: Input Validation
def test_input_validation():
    """Test that invalid inputs are handled gracefully"""
    # Empty transcript should use fallback
    # (This would need mock of LLM)
    pass

# Test 4: Logging
def test_logging_active():
    """Test that logging is properly configured"""
    import logging
    logger = logging.getLogger('memory_store')
    assert logger.level >= logging.DEBUG

if __name__ == "__main__":
    test_memory_fallback_mode()
    test_rate_limiting()
    test_logging_active()
    print("✅ All tests passed!")
```

---

## Integration Tests (Local)

### Test 1: Memory System Fallback
```bash
# Stop Qdrant
docker stop qdrant  # or equivalent

# Run memory test
python -c "
from realtime_conversational_ai import ConversationalAI
ca = ConversationalAI()
result = ca.test_memory_pipeline()
print('Memory stats:', result['memory_stats'])
"

# Check logs for:
# ✅ "⚠️ Memory store initialization failed"
# ✅ "Entering fallback mode"
# Expected: Tests still run, memory returns empty results
```

### Test 2: Input Validation
```bash
# Test empty audio
python -c "
import base64
empty_audio = base64.b64encode(b'').decode('utf-8')
# (Would need to mock conversation process)
# Expected: Log shows "Empty/invalid transcript, using fallback"
"
```

### Test 3: Rate Limiting
```bash
# Test using curl
for i in {1..5}; do
  curl -X POST http://localhost:8000/ws/conversation \
    -H "Content-Type: application/json" \
    -d '{"session_id":"test","audio":"data"}'
done

# Expected: Requests 2-5 return rate limit error
```

### Test 4: Health Endpoints
```bash
# Check all health endpoints
curl http://localhost:8000/health
curl http://localhost:8000/readiness
curl http://localhost:8000/liveness

# Expected: All return 200
# Check readiness reports memory status correctly
```

---

## Modal Deployment Tests

### Pre-Deployment
```bash
# Verify Modal CLI
modal version

# Test locally with Modal
modal run realtime_conversational_ai.py

# Check logs
modal logs realtime_conversational_ai --tail 100
```

### Post-Deployment
```bash
# Test deployed endpoints
DEPLOY_URL="https://your-app.modal.run"

# Test health
curl $DEPLOY_URL/health
# Expected: {"status": "healthy"}

# Test readiness
curl $DEPLOY_URL/readiness
# Expected: {"status": "ready", "memory": "available"/"unavailable"}

# Test liveness
curl $DEPLOY_URL/liveness
# Expected: {"status": "alive"}

# Test WebSocket (needs client implementation)
# See client test below
```

---

## Client Testing (Browser/JavaScript)

Create `test_client.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Test Client</title>
    <script>
        async function testWebSocket() {
            const sessionId = "test_" + Date.now();
            const ws = new WebSocket("wss://your-app.modal.run/ws/conversation");
            
            ws.onopen = () => {
                console.log("✅ WebSocket connected");
                
                // Get fake audio data
                const audioData = "data:audio/wav;base64,...";  // Replace with real audio
                
                ws.send(JSON.stringify({
                    session_id: sessionId,
                    audio: audioData,
                    voice_name: "en-US-DragonV2.1Neural"
                }));
            };
            
            ws.onmessage = (event) => {
                const response = JSON.parse(event.data);
                console.log("✅ Got response:", response);
                
                if (response.llm_reply) console.log("💬 LLM:", response.llm_reply);
                if (response.tts_audio) console.log("🎵 Audio length:", response.tts_audio.length);
                if (response.error) console.log("❌ Error:", response.error);
                
                ws.close();
            };
            
            ws.onerror = (error) => {
                console.error("❌ WebSocket error:", error);
            };
            
            ws.onclose = (event) => {
                console.log("ℹ️ WebSocket closed");
                console.log("  Code:", event.code);
                console.log("  Reason:", event.reason);
                
                // Interpret close codes
                if (event.code === 1008) console.log("  → Policy violation (client issue)");
                if (event.code === 1011) console.log("  → Server error (retry needed)");
            };
        }
        
        // Test rate limiting
        async function testRateLimit() {
            const sessionId = "rate_test";
            const ws = new WebSocket("wss://your-app.modal.run/ws/conversation");
            
            let sent = 0;
            ws.onopen = () => {
                for (let i = 0; i < 3; i++) {
                    setTimeout(() => {
                        ws.send(JSON.stringify({
                            session_id: sessionId,
                            audio: "fake_audio"
                        }));
                        sent++;
                    }, i * 100);  // Rapid fire
                }
            };
            
            ws.onmessage = (event) => {
                const response = JSON.parse(event.data);
                if (response.error === "Too Many Requests") {
                    console.log("✅ Rate limiting working!");
                }
            };
        }
    </script>
</head>
<body>
    <h1>Production Hardening Tests</h1>
    <button onclick="testWebSocket()">Test WebSocket</button>
    <button onclick="testRateLimit()">Test Rate Limiting</button>
    <pre id="output"></pre>
</body>
</html>
```

---

## Stress Testing

### Test 1: High Load
```bash
# Use Apache Bench or similar
ab -n 100 -c 10 http://localhost:8000/health

# Expected: All requests successful within 500ms
```

### Test 2: Concurrent Sessions
```bash
# Create 50 concurrent WebSocket connections
# Send conversation turns
# Expected:
# - No race condition errors
# - All sessions maintain separate state
# - No session data mixing
```

### Test 3: Memory Under Load
```bash
# Run for 10 minutes with continuous requests
# Monitor:
# - Memory growth (should be stable)
# - Session count (should cleanup old sessions)
# - Error rate (should remain <0.1%)
```

---

## Expected Results Summary

### ✅ Should See
- All syntax checks pass
- `logger` calls instead of `print()` in logs
- Health endpoints return 200
- Readiness shows memory: "available"/"unavailable"
- Rate limiting triggers on rapid requests
- WebSocket close codes (1008, 1011)
- Meaningful error messages
- No stack traces to client (truncated)
- Qdrant fallback mode activates when unavailable
- Session locks prevent race conditions

### ❌ Should NOT See
- Any `print()` output (use logging instead)
- Silent crashes or hung requests
- Empty error messages
- Protocol errors in network tab
- Memory growing unbounded
- Sessions interfering with each other
- Unhandled exceptions in logs

---

## Validation Checklist

- [ ] No syntax errors: `python -m py_compile realtime_conversational_ai.py`
- [ ] Imports work: `python -c "from realtime_conversational_ai import *"`
- [ ] Health endpoint returns 200
- [ ] Readiness endpoint shows memory status
- [ ] Liveness endpoint returns 200
- [ ] Rate limiting works (3rd request fails)
- [ ] WebSocket errors have meaningful close codes
- [ ] Logs show structured format with timestamps
- [ ] Fallback mode activates when Qdrant down
- [ ] No sensitive data in error messages
- [ ] Session state is thread-safe
- [ ] All 9 improvements verified

---

## Continuous Validation in Production

Monitor these metrics after deployment:

```
✅ Daily Check:
   - Error rate < 0.1%
   - Response latency P95 < 10s
   - 0 "CRASH" errors in logs
   - 0 unhandled exceptions

✅ Weekly Check:
   - Memory system availability > 99%
   - Rate limiting protecting against abuse
   - No sessions in inconsistent state
   - Cold start time < 25s

✅ Monthly Check:
   - Review all error patterns
   - Check for memory leaks
   - Validate graceful degradation still works
   - Audit security (rate limits, data isolation)
```

---

## Common Issues & Diagnostics

### Issue: "Qdrant unavailable" logs but system working

**Diagnosis:** ✅ Expected - fallback mode active  
**Check:** `GET /readiness` should show `{"memory": "unavailable"}`  
**Action:** Verify Qdrant URL and connectivity

### Issue: Requests timing out

**Diagnosis:** Check component latencies  
```bash
modal logs realtime_conversational_ai | grep "ms"
# Look for: asr_latency, emotion_latency, llm_latency, tts_latency
```
**Action:** Optimize slowest component

### Issue: Memory state corrupted/mixed

**Diagnosis:** Session locks not working  
**Check:** Verify asyncio.Lock is being used  
**Action:** Check for await points outside locks

### Issue: Rate limiting too aggressive

**Diagnosis:** Users can't send requests  
**Check:** Adjust `rate_limit_window` (currently 2.0s)  
**Action:** Increase window and redeploy

---

**Testing Status:** Ready for comprehensive validation  
**Expected Result:** All checks should pass ✅
