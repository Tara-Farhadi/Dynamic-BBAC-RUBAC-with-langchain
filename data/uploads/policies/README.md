# Policy Documents Upload Folder

Place your policy PDF files here.

## Policy Types

### 1. Organizational Policies
Company-specific rules and guidelines:
- Transaction amount limits
- Geographic restrictions
- Time-based rules (e.g., no large transactions after midnight)
- Velocity limits (e.g., max 3 transactions per hour)
- Merchant category restrictions
- Emergency access protocols

### 2. Regulatory Policies
Government and industry regulations:
- GDPR compliance requirements
- PCI-DSS standards
- Country-specific financial regulations
- Sanctions lists (e.g., OFAC)
- Data protection laws
- Cross-border transaction restrictions

## File Format

- **Format**: PDF only
- **Naming**: Use descriptive names (e.g., `transaction_limits.pdf`, `ofac_sanctions.pdf`)
- **Content**: Text-based PDFs (not scanned images)
- **Size**: Any size supported
- **Language**: English

## Example Policy Files

You can generate sample policy files:
```bash
python generate_sample_policies.py
```

This creates:
- `organizational_policy.txt` - Convert to PDF
- `regulatory_policy.txt` - Convert to PDF

## How to Upload

### Option 1: API Upload (Organizational)
```bash
curl -X POST "http://localhost:8000/api/v1/policies?policy_type=organizational" \
  -F "file=@C:/nastaran/guardian-system/data/uploads/policies/transaction_limits.pdf"
```

### Option 2: API Upload (Regulatory)
```bash
curl -X POST "http://localhost:8000/api/v1/policies?policy_type=regulatory" \
  -F "file=@C:/nastaran/guardian-system/data/uploads/policies/ofac_sanctions.pdf"
```

### Option 3: Use Swagger UI
1. Go to http://localhost:8000/docs
2. Find `/api/v1/policies` endpoint
3. Click "Try it out"
4. Select policy_type (organizational or regulatory)
5. Upload your PDF file
6. Click "Execute"

## Policy Processing

When you upload a policy PDF:
1. System extracts text from PDF
2. Chunks text into 300-500 token segments
3. Generates embeddings for each chunk
4. Stores in vector database for RAG queries
5. Policy becomes immediately active

## Policy Update

To update a policy:
1. Upload the new version with the same or different filename
2. System automatically indexes the new content
3. Old and new policies both remain searchable
4. You can manage policy status in the database

## Tips

- Keep policies organized by category
- Use clear, descriptive filenames
- Update policies regularly to reflect new rules
- Test policy enforcement after upload using evaluation endpoint
