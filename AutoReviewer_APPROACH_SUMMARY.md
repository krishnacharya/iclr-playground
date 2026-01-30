# AutoReviewer Approach Summary

## Key Findings from AutoReviewer Codebase

### 1. API Client Usage

**For 2020-2023 (API v1):**
```python
import openreview
client = openreview.Client(baseurl='https://api.openreview.net')
submissions = client.get_all_notes(
    invitation='ICLR.cc/2020/Conference/-/Blind_Submission',
    details='replies'  # or 'directReplies'
)
```

**For 2024+ (API v2):**
```python
import openreview.api
client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
submissions = client.get_all_notes(
    invitation='ICLR.cc/2024/Conference/-/Submission',
    details='replies'
)
```

### 2. Review Extraction

**Key differences:**
- **API v1 (2020-2023)**: Uses `invitation` field (string)
- **API v2 (2024+)**: Uses `invitations` field (array)

**Extracting reviews:**
```python
# For API v1
for reply in submission.details.get('replies', []):
    invitation = reply.get('invitation', '')
    if 'Official_Review' in invitation:
        # This is a review
        content = reply.get('content', {})
        rating = content.get('rating', '')
        review_text = content.get('review', '')

# For API v2
for reply in submission.details.get('replies', []):
    invitations = reply.get('invitations', [])
    if any('Official_Review' in inv for inv in invitations):
        # This is a review
        content = reply.get('content', {})
        # API v2 uses nested format: {'rating': {'value': '...'}}
        rating = content.get('rating', {})
        if isinstance(rating, dict):
            rating = rating.get('value', '')
```

### 3. PDF Downloads

**API v1:**
```python
pdf_bytes = client.get_pdf(submission_id)
# Or for revisions:
pdf_bytes = client.get_pdf(revision_id, is_reference=True)
```

**API v2:**
```python
pdf_bytes = client.get_pdf(submission_id)
# Or for edits:
url = client.baseurl + '/notes/edits/attachment'
params = {'id': edit_id, 'name': 'pdf'}
response = client.session.get(url, params=params)
pdf_bytes = response.content
```

### 4. Handling API v2 Nested Values

API v2 wraps values in `{'value': ...}` format:
```python
def get_value_api2(field):
    """Extract value from API v2 nested dict format."""
    if isinstance(field, dict):
        return field.get('value', field)
    return field

# Usage:
title = get_value_api2(submission.content.get('title', ''))
abstract = get_value_api2(submission.content.get('abstract', ''))
```

### 5. Review Content Structure (2020)

From the API response, reviews have:
- `rating`: String like "1: Reject" or "3: Weak Reject"
- `review`: Full review text
- `title`: Review title
- `experience_assessment`: Reviewer expertise
- `review_assessment:_*`: Various assessment fields

### 6. Decision Extraction

**API v1:**
```python
for reply in submission.details.get('replies', []):
    if 'Decision' in reply.get('invitation', ''):
        decision = reply.get('content', {}).get('decision', '')
```

**API v2:**
```python
for reply in submission.details.get('replies', []):
    if any('Decision' in inv for inv in reply.get('invitations', [])):
        decision = reply.get('content', {}).get('decision', {})
        if isinstance(decision, dict):
            decision = decision.get('value', '')
```

### 7. Rate Limiting

AutoReviewer uses:
- Small delays between requests (`time.sleep(0.5)`)
- Retry logic with exponential backoff for 429 errors
- Proper error handling

### 8. Best Practices

1. **Always use `details='replies'`** when fetching submissions to get full reply tree
2. **Handle both API versions** - check year and use appropriate client
3. **Extract nested values** for API v2 using helper function
4. **Check invitation types** to filter reviews vs comments vs decisions
5. **Use forum ID** from submission to fetch all related notes
6. **Handle rate limiting** gracefully with retries
