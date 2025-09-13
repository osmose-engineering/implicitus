export async function postJsonWithSid(base: string, path: string, body: any, sid?: string) {
  const url = new URL(path, base);
  if (sid) {
    url.searchParams.set('sid', sid);
  }
  const response = await fetch(url.toString(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status} ${response.statusText} - ${text}`);
  }
  return await response.json();
}
