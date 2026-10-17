const API_BASE = import.meta.env.VITE_API_URL ?? '/api'

export interface SdsDocument {
  filename: string
  sections: number
}

export interface Source {
  document: string
  section: number
  title: string
}

export interface ChatResponse {
  answer: string
  sources: Source[]
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, init)
  } catch {
    throw new Error('Could not reach the SDS service. Start the backend and try again.')
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    const detail = body?.detail
    const message = typeof detail === 'string'
      ? detail
      : Array.isArray(detail)
        ? detail.map((item: { msg?: string }) => item.msg).filter(Boolean).join(' ')
        : `Request failed (${response.status})`
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string; chunks_stored: number }>('/'),
  documents: () => request<{ documents: SdsDocument[]; total_sections: number }>('/documents'),
  chat: (question: string) => {
    return request<ChatResponse>('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })
  },
  upload: (file: File) => {
    const body = new FormData()
    body.append('file', file)
    return request<{ filename: string; sections_found: number; message: string }>('/upload', {
      method: 'POST', body,
    })
  },
}
