-- name: GetAppDocument :one
SELECT document_key, payload, source_path, updated_at
FROM app_documents
WHERE document_key = $1;

-- name: UpsertAppDocument :exec
INSERT INTO app_documents (document_key, payload, source_path, updated_at)
VALUES ($1, $2, $3, NOW())
ON CONFLICT (document_key) DO UPDATE SET
    payload = EXCLUDED.payload,
    source_path = COALESCE(EXCLUDED.source_path, app_documents.source_path),
    updated_at = NOW();
