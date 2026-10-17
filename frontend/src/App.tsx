import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { Activity, ArrowDown, ArrowUp, Check, ChevronDown, FileText, FlaskConical, FolderOpen, LoaderCircle, Menu, MessageSquareText, Paperclip, Plus, Search, ShieldCheck, Sparkles, X } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { api, type ChatResponse, type SdsDocument, type Source } from './lib/api'

type Message = { id: string; role: 'user' | 'assistant'; content: string; sources?: Source[]; pending?: boolean; error?: boolean }
const suggestions = [
  { icon: ShieldCheck, label: 'First aid', question: 'What first aid measures should I take after exposure?' },
  { icon: Activity, label: 'Handling & storage', question: 'What are the safe handling and storage requirements?' },
  { icon: FlaskConical, label: 'Chemical properties', question: 'What are the physical and chemical properties?' },
]
const formatName = (filename: string) => filename.replace(/\.pdf$/i, '').replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())

function App() {
  const [documents, setDocuments] = useState<SdsDocument[]>([])
  const [totalSections, setTotalSections] = useState(0)
  const [connected, setConnected] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadNotice, setUploadNotice] = useState('')
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')
  const [mobileNav, setMobileNav] = useState(false)
  const [activeDocument, setActiveDocument] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const refresh = useCallback(async () => {
    try {
      const [health, inventory] = await Promise.all([api.health(), api.documents()])
      setConnected(health.status === 'running')
      setDocuments(inventory.documents)
      setTotalSections(inventory.total_sections)
    } catch { setConnected(false) }
  }, [])

  useEffect(() => { void refresh(); const timer = window.setInterval(() => void refresh(), 30_000); return () => window.clearInterval(timer) }, [refresh])
  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight }, [messages, busy])
  useEffect(() => { if (textareaRef.current) { textareaRef.current.style.height = 'auto'; textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px` } }, [question])

  async function ask(raw = question) {
    const value = raw.trim()
    if (!value || busy) return
    setError(''); setQuestion(''); setBusy(true)
    const userMessage: Message = { id: crypto.randomUUID(), role: 'user', content: value }
    const pending: Message = { id: crypto.randomUUID(), role: 'assistant', content: '', pending: true }
    setMessages((current) => [...current, userMessage, pending])
    try {
      const result: ChatResponse = await api.chat(value)
      setMessages((current) => current.map((item) => item.id === pending.id ? { ...item, content: result.answer, sources: result.sources, pending: false } : item))
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : 'Something went wrong.'
      setMessages((current) => current.map((item) => item.id === pending.id ? { ...item, content: message, pending: false, error: true } : item))
    } finally { setBusy(false); textareaRef.current?.focus() }
  }

  async function onSubmit(event: FormEvent) { event.preventDefault(); await ask() }
  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void ask() } }
  async function onFile(file?: File) {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.pdf')) { setError('Choose a PDF file to add it to your library.'); return }
    setUploading(true); setError(''); setUploadNotice('')
    try {
      const result = await api.upload(file)
      setUploadNotice(`${formatName(result.filename)} added · ${result.sections_found} sections indexed`)
      await refresh()
      window.setTimeout(() => setUploadNotice(''), 6500)
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not upload this PDF.') }
    finally { setUploading(false); if (fileInput.current) fileInput.current.value = '' }
  }

  const filteredDocuments = documents.filter((document) => formatName(document.filename).toLowerCase().includes(search.toLowerCase()))

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`}>
        <div className="brand-row"><div className="brand-mark"><FlaskConical size={19} strokeWidth={2.1} /></div><div className="brand-name">fieldnote<span>.</span><small>SAFETY INTELLIGENCE</small></div><button className="icon-button mobile-close" onClick={() => setMobileNav(false)} aria-label="Close navigation"><X size={17} /></button></div>
        <div className="workspace-switch"><div className="workspace-icon">N</div><div className="workspace-copy"><strong>Northstar Labs</strong><span>Research workspace</span></div><ChevronDown size={15} className="muted-icon" /></div>
        <button className="new-chat" onClick={() => { setMessages([]); setQuestion(''); setError(''); setMobileNav(false) }}><Plus size={17} /><span>New conversation</span><kbd>⌘ K</kbd></button>
        <div className="nav-caption">WORKSPACE</div>
        <button className="nav-item nav-item-active"><MessageSquareText size={16} /><span>Ask your library</span></button>
        <button className="nav-item" onClick={() => fileInput.current?.click()}><FolderOpen size={16} /><span>Document library</span><span className="nav-count">{documents.length}</span></button>
        <div className="library-heading"><div className="nav-caption">YOUR LIBRARY <span>{documents.length}</span></div><button className="icon-button tiny" onClick={() => fileInput.current?.click()} aria-label="Add a document"><Plus size={16} /></button></div>
        <label className="search-box"><Search size={14} /><input placeholder="Find a document" value={search} onChange={(e) => setSearch(e.target.value)} />{search && <button onClick={() => setSearch('')} aria-label="Clear search"><X size={13} /></button>}</label>
        <div className="document-list">
          {filteredDocuments.map((document) => <button className={`document-item ${activeDocument === document.filename ? 'document-active' : ''}`} key={document.filename} onClick={() => { setActiveDocument(document.filename); setQuestion(`Summarize the key safety information in ${formatName(document.filename)}.`); setMobileNav(false); textareaRef.current?.focus() }}><span className="pdf-icon"><FileText size={14} /></span><span className="document-copy"><strong>{formatName(document.filename)}</strong><small>{document.sections} sections</small></span><ChevronDown className="document-chevron" size={14} /></button>)}
          {filteredDocuments.length === 0 && <div className="empty-documents">{search ? 'No matching documents' : 'Your library is empty'}</div>}
        </div>
        <button className="add-document" onClick={() => fileInput.current?.click()} disabled={uploading}>{uploading ? <LoaderCircle className="spin" size={15} /> : <Plus size={15} />}<span>{uploading ? 'Indexing document…' : 'Add a document'}</span></button>
        <div className="sidebar-bottom"><div className="indexed-card"><div className="indexed-head"><span className="indexed-dot" /><span>LIBRARY STATUS</span><span className="indexed-live">{connected ? 'LIVE' : 'OFFLINE'}</span></div><div className="indexed-number">{totalSections.toLocaleString()}<span> sections indexed</span></div><div className="indexed-bar"><i style={{ width: `${Math.min(totalSections / 160, 100)}%` }} /></div><div className="indexed-foot"><span>{documents.length} documents</span><span>Gemini connected</span></div></div><div className="profile-row"><div className="avatar">S</div><div className="profile-copy"><strong>Sam Patel</strong><span>Personal workspace</span></div><button className="icon-button tiny" aria-label="Profile menu"><ChevronDown size={15} /></button></div></div>
      </aside>
      {mobileNav && <button className="mobile-scrim" onClick={() => setMobileNav(false)} aria-label="Close navigation" />}
      <main className="main-area">
        <header className="topbar"><div className="topbar-left"><button className="icon-button mobile-menu" onClick={() => setMobileNav(true)} aria-label="Open navigation"><Menu size={19} /></button><div className="breadcrumb">Workspace <span>/</span> <strong>Ask your library</strong></div></div><div className="topbar-right"><span className={`service-state ${connected ? '' : 'state-offline'}`}><i />{connected ? 'All systems operational' : 'Service unavailable'}</span><span className="topbar-divider" /><button className="help-button" onClick={() => { setQuestion('How do I use SDS Intelligence?'); textareaRef.current?.focus() }}>Help &amp; feedback <span>↗</span></button></div></header>
        <section className={`conversation ${messages.length ? 'conversation-active' : ''}`} ref={scrollRef}>
          {messages.length === 0 ? <div className="welcome-wrap"><div className="welcome-eyebrow"><span className="eyebrow-line" /> YOUR CHEMICAL SAFETY DESK <span className="eyebrow-line" /></div><div className="welcome-icon"><Sparkles size={20} strokeWidth={1.8} /></div><h1>Good morning, Sam<span>.</span></h1><p className="welcome-subtitle">Your safety documents, ready to answer.</p><p className="welcome-body">Ask a question across your SDS library. Every answer is grounded in your documents, with sources you can verify.</p><div className="suggestion-grid">{suggestions.map(({ icon: Icon, label, question: prompt }) => <button key={label} className="suggestion-card" onClick={() => void ask(prompt)}><span className="suggestion-icon"><Icon size={16} /></span><span><strong>{label}</strong><small>{prompt}</small></span><ArrowDown className="suggestion-arrow" size={14} /></button>)}</div><div className="grounding-note"><ShieldCheck size={15} /><span>Answers are grounded in your documents and always include their source.</span></div></div> : <div className="message-list">{messages.map((message) => <article className={`message message-${message.role}`} key={message.id}><div className={`message-avatar ${message.role === 'assistant' ? 'assistant-avatar' : ''}`}>{message.role === 'assistant' ? <FlaskConical size={15} /> : 'S'}</div><div className="message-content"><div className="message-meta"><strong>{message.role === 'assistant' ? 'Fieldnote' : 'You'}</strong><span>{message.pending ? 'Searching your library' : message.role === 'assistant' ? 'AI response' : 'Just now'}</span></div>{message.pending ? <div className="thinking"><span /><span /><span /><span>Reviewing relevant sections…</span></div> : <div className={`markdown-body ${message.error ? 'error-copy' : ''}`}><ReactMarkdown>{message.content}</ReactMarkdown></div>}{message.sources && message.sources.length > 0 && <div className="sources"><div className="sources-label"><Check size={13} /> SOURCES IN YOUR LIBRARY</div><div className="source-list">{[...new Map(message.sources.map((source) => [`${source.document}-${source.section}`, source])).values()].map((source) => <div className="source-chip" key={`${source.document}-${source.section}`}><FileText size={13} /><span>{formatName(source.document)}</span><i>·</i><span>Section {source.section}</span><small>{source.title}</small></div>)}</div></div>}</div></article>)}</div>}
        </section>
        <div className="composer-region"><div className="composer-wrap">{error && <div className="error-banner"><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={15} /></button></div>}{uploadNotice && <div className="success-banner"><Check size={15} /><span>{uploadNotice}</span><button onClick={() => setUploadNotice('')} aria-label="Dismiss notification"><X size={15} /></button></div>}<form className="composer" onSubmit={onSubmit}><textarea ref={textareaRef} value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={onKeyDown} placeholder="Ask anything about your safety documents…" rows={1} aria-label="Ask a safety question" /><div className="composer-tools"><div className="composer-hints"><button type="button" className="attach-button" onClick={() => fileInput.current?.click()} aria-label="Attach a PDF"><Paperclip size={15} /></button><span className="composer-separator" /><span className="grounded-label"><ShieldCheck size={13} /> Grounded in your library</span></div><div className="composer-submit"><span className="keyboard-hint"><kbd>↵</kbd> to send</span><button className="send-button" type="submit" disabled={!question.trim() || busy} aria-label="Send message">{busy ? <LoaderCircle className="spin" size={16} /> : <ArrowUp size={17} strokeWidth={2.1} />}</button></div></div></form><div className="disclaimer">AI-generated answers may be incomplete. Always refer to the original SDS for critical safety decisions.</div></div><footer className="footer"><span>FIELDNOTE <i>·</i> SAFETY INTELLIGENCE</span><span><span className="footer-dot" /> PRIVATE WORKSPACE</span></footer></div>
      </main>
      <input ref={fileInput} className="hidden-input" type="file" accept="application/pdf,.pdf" onChange={(event) => void onFile(event.target.files?.[0])} />
    </div>
  )
}

export default App
