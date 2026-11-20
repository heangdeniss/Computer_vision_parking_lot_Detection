/** POST a FormData body and unwrap the API's error envelope. */
export async function postForm(path, body) {
  const response = await fetch(path, { method: 'POST', body })
  const data = await response.json()
  if (!response.ok) throw new Error(data.error || 'request failed')
  return data
}
