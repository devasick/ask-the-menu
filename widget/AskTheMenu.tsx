/**
 * Ask the Menu chat widget: a floating button that opens a chat panel for one restaurant.
 *
 *   <AskTheMenu apiUrl="https://ask.example.com" restaurantId="…" widgetKey="…" language="de" />
 *
 * React 18+, no UI library, styles scoped under `.atm-`. Mobile first: full screen on phones,
 * a floating card from 640px up. The widget key is public by design (it only allows /chat for
 * this restaurant), so it can ship in the page.
 */
import { useEffect, useId, useRef, useState, type FormEvent } from "react";

export type Language = "en" | "de";

export interface AskTheMenuProps {
  apiUrl: string;
  restaurantId: string;
  widgetKey: string;
  language?: Language;
}

/** Mirrors ChatResponse / ItemCard in app/api/chat.py. */
interface ItemCard {
  id: string;
  name: string;
  price: string; // decimal as string, e.g. "15.90"
  currency: string;
  allergens: string[] | null; // null = unknown, ask staff
  dietary_tags: string[];
  available: boolean;
}

interface ChatResponse {
  answer: string;
  items: ItemCard[];
  route: string;
  sources: { type: string; id: string }[];
  disclaimer: string | null;
}

type Message =
  | { role: "guest"; text: string }
  | { role: "assistant"; reply: ChatResponse }
  | { role: "error"; text: string };

const TEXT = {
  en: {
    open: "Ask the menu",
    close: "Close chat",
    title: "Ask the menu",
    intro: "Ask about dishes, prices, allergens or opening hours.",
    placeholder: "e.g. What's vegan under €15?",
    send: "Send",
    thinking: "Looking at the menu…",
    error: "Sorry, something went wrong. Please try again or ask the staff.",
    allergens: "Allergens",
    allergensUnknown: "Allergens unknown, please ask staff",
    soldOut: "Sold out today",
    tooLong: "Please keep your question under 500 characters.",
  },
  de: {
    open: "Frag die Karte",
    close: "Chat schließen",
    title: "Frag die Karte",
    intro: "Fragen Sie nach Gerichten, Preisen, Allergenen oder Öffnungszeiten.",
    placeholder: "z. B. Was ist vegan unter 15 €?",
    send: "Senden",
    thinking: "Ich schaue in die Karte…",
    error: "Leider ist etwas schiefgelaufen. Bitte versuchen Sie es erneut oder fragen Sie das Personal.",
    allergens: "Allergene",
    allergensUnknown: "Allergene unbekannt, bitte Personal fragen",
    soldOut: "Heute ausverkauft",
    tooLong: "Bitte stellen Sie Ihre Frage in weniger als 500 Zeichen.",
  },
} as const;

const ALLERGEN_NAMES: Record<Language, Record<string, string>> = {
  en: {
    gluten: "gluten", crustaceans: "crustaceans", eggs: "eggs", fish: "fish",
    peanuts: "peanuts", soybeans: "soy", milk: "milk", nuts: "nuts", celery: "celery",
    mustard: "mustard", sesame: "sesame", sulphites: "sulphites", lupin: "lupin",
    molluscs: "molluscs",
  },
  de: {
    gluten: "Gluten", crustaceans: "Krebstiere", eggs: "Eier", fish: "Fisch",
    peanuts: "Erdnüsse", soybeans: "Soja", milk: "Milch", nuts: "Schalenfrüchte",
    celery: "Sellerie", mustard: "Senf", sesame: "Sesam", sulphites: "Sulfite",
    lupin: "Lupinen", molluscs: "Weichtiere",
  },
};

const MAX_QUESTION = 500;

function sessionId(): string {
  const key = "atm-session";
  try {
    const existing = sessionStorage.getItem(key);
    if (existing) return existing;
    const id = crypto.randomUUID();
    sessionStorage.setItem(key, id);
    return id;
  } catch {
    return crypto.randomUUID(); // storage blocked (private mode): one id per page load
  }
}

function formatPrice(item: ItemCard, language: Language): string {
  const locale = language === "de" ? "de-DE" : "en-GB";
  return new Intl.NumberFormat(locale, { style: "currency", currency: item.currency }).format(
    Number(item.price),
  );
}

export function AskTheMenu({ apiUrl, restaurantId, widgetKey, language = "de" }: AskTheMenuProps) {
  const t = TEXT[language];
  const [open, setOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const panelId = useId();
  const titleId = useId();
  const buttonRef = useRef<HTMLButtonElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const session = useRef<string>("");

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [messages, busy]);

  function close() {
    setOpen(false);
    // The button is hidden while the panel is open; focus it once it is visible again.
    requestAnimationFrame(() => buttonRef.current?.focus());
  }

  async function ask(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || busy) return;
    if (text.length > MAX_QUESTION) {
      setMessages((m) => [...m, { role: "error", text: t.tooLong }]);
      return;
    }
    session.current ||= sessionId();
    setQuestion("");
    setMessages((m) => [...m, { role: "guest", text }]);
    setBusy(true);
    try {
      const response = await fetch(`${apiUrl.replace(/\/$/, "")}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Widget-Key": widgetKey },
        body: JSON.stringify({
          restaurant_id: restaurantId,
          session_id: session.current,
          question: text,
          language,
        }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const reply = (await response.json()) as ChatResponse;
      setMessages((m) => [...m, { role: "assistant", reply }]);
    } catch {
      setMessages((m) => [...m, { role: "error", text: t.error }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="atm-root" lang={language}>
      <style>{CSS}</style>
      <button
        ref={buttonRef}
        type="button"
        className="atm-fab"
        aria-expanded={open}
        aria-controls={panelId}
        hidden={open}
        onClick={() => setOpen(true)}
      >
        <span aria-hidden="true">💬</span> {t.open}
      </button>
      {open && (
        <section
          id={panelId}
          className="atm-panel"
          role="dialog"
          aria-labelledby={titleId}
          onKeyDown={(e) => e.key === "Escape" && close()}
        >
          <header className="atm-header">
            <h2 id={titleId}>{t.title}</h2>
            <button type="button" className="atm-close" aria-label={t.close} onClick={close}>
              ×
            </button>
          </header>
          <div ref={logRef} className="atm-log" role="log" aria-live="polite" aria-busy={busy}>
            <p className="atm-intro">{t.intro}</p>
            {messages.map((m, i) => (
              <MessageView key={i} message={m} language={language} />
            ))}
            {busy && <p className="atm-thinking">{t.thinking}</p>}
          </div>
          <form className="atm-form" onSubmit={ask}>
            <label className="atm-visually-hidden" htmlFor={`${panelId}-q`}>
              {t.placeholder}
            </label>
            <input
              ref={inputRef}
              id={`${panelId}-q`}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={t.placeholder}
              maxLength={MAX_QUESTION}
              autoComplete="off"
              enterKeyHint="send"
            />
            <button type="submit" disabled={busy || !question.trim()}>
              {t.send}
            </button>
          </form>
        </section>
      )}
    </div>
  );
}

function MessageView({ message, language }: { message: Message; language: Language }) {
  if (message.role === "guest") return <p className="atm-msg atm-guest">{message.text}</p>;
  if (message.role === "error")
    return (
      <p className="atm-msg atm-error" role="alert">
        {message.text}
      </p>
    );
  const { reply } = message;
  return (
    <div className="atm-msg atm-assistant">
      <p>{reply.answer}</p>
      {reply.items.length > 0 && (
        <ul className="atm-cards">
          {reply.items.map((item) => (
            <ItemCardView key={item.id} item={item} language={language} />
          ))}
        </ul>
      )}
      {reply.disclaimer && <p className="atm-disclaimer">{reply.disclaimer}</p>}
    </div>
  );
}

function ItemCardView({ item, language }: { item: ItemCard; language: Language }) {
  const t = TEXT[language];
  const names = ALLERGEN_NAMES[language];
  return (
    <li className={`atm-card${item.available ? "" : " atm-sold-out"}`}>
      <div className="atm-card-head">
        <span className="atm-name">{item.name}</span>
        <span className="atm-price">{formatPrice(item, language)}</span>
      </div>
      {!item.available && <span className="atm-badge atm-badge-warn">{t.soldOut}</span>}
      <div className="atm-badges" aria-label={t.allergens}>
        {/* The API sends null for unknown; an empty list is treated the same, never as safe. */}
        {!item.allergens?.length ? (
          <span className="atm-badge atm-badge-warn">{t.allergensUnknown}</span>
        ) : (
          item.allergens.map((a) => (
            <span key={a} className="atm-badge atm-badge-allergen">
              {names[a] ?? a}
            </span>
          ))
        )}
      </div>
    </li>
  );
}

const CSS = `
.atm-root { --atm-accent: #b4401c; --atm-bg: #fff; --atm-fg: #1d1d1b; --atm-muted: #5f5f5a;
  --atm-line: #e3e1db; --atm-warn-bg: #fff3cd; --atm-warn-fg: #6b4e00;
  font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--atm-fg); }
@media (prefers-color-scheme: dark) { .atm-root { --atm-bg: #1f1f1d; --atm-fg: #f2f1ed;
  --atm-muted: #b3b1aa; --atm-line: #3a3a36; --atm-warn-bg: #4a3b10; --atm-warn-fg: #ffe08a; } }
.atm-root * { box-sizing: border-box; }
.atm-fab { position: fixed; right: 16px; bottom: 16px; z-index: 2147483000; min-height: 48px;
  padding: 0 20px; border: 0; border-radius: 24px; background: var(--atm-accent); color: #fff;
  font: inherit; font-weight: 600; box-shadow: 0 4px 14px rgb(0 0 0 / .25); cursor: pointer; }
.atm-panel { position: fixed; inset: 0; z-index: 2147483000; display: flex; flex-direction: column;
  background: var(--atm-bg); }
@media (min-width: 640px) { .atm-panel { inset: auto 16px 16px auto; width: 380px; height: min(620px, calc(100vh - 32px));
  border: 1px solid var(--atm-line); border-radius: 16px; box-shadow: 0 10px 40px rgb(0 0 0 / .25); overflow: hidden; } }
.atm-header { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px;
  border-bottom: 1px solid var(--atm-line); }
.atm-header h2 { margin: 0; font-size: 17px; }
.atm-close { width: 44px; height: 44px; border: 0; background: none; color: inherit; font-size: 26px; cursor: pointer; }
.atm-log { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
.atm-intro, .atm-thinking { margin: 0; color: var(--atm-muted); }
.atm-msg { margin: 0; max-width: 92%; padding: 10px 12px; border-radius: 12px; }
.atm-msg p { margin: 0; }
.atm-guest { align-self: flex-end; background: var(--atm-accent); color: #fff; }
.atm-assistant { align-self: flex-start; border: 1px solid var(--atm-line); }
.atm-error { align-self: flex-start; background: var(--atm-warn-bg); color: var(--atm-warn-fg); }
.atm-cards { list-style: none; margin: 10px 0 0; padding: 0; display: grid; gap: 8px; }
.atm-card { padding: 10px; border: 1px solid var(--atm-line); border-radius: 10px; }
.atm-sold-out { opacity: .7; }
.atm-card-head { display: flex; justify-content: space-between; gap: 8px; font-weight: 600; }
.atm-price { white-space: nowrap; }
.atm-badges { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.atm-badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px;
  border: 1px solid var(--atm-line); color: var(--atm-muted); }
.atm-badge-allergen { border-color: var(--atm-accent); color: var(--atm-accent); }
.atm-badge-warn { background: var(--atm-warn-bg); color: var(--atm-warn-fg); border-color: transparent; margin-top: 6px; }
.atm-disclaimer { margin-top: 8px !important; font-size: 12px; color: var(--atm-muted); }
.atm-form { display: flex; gap: 8px; padding: 12px; border-top: 1px solid var(--atm-line);
  padding-bottom: max(12px, env(safe-area-inset-bottom)); }
.atm-form input { flex: 1; min-height: 44px; padding: 0 12px; border: 1px solid var(--atm-line);
  border-radius: 10px; background: transparent; color: inherit; font-size: 16px; }
.atm-form button { min-height: 44px; padding: 0 16px; border: 0; border-radius: 10px;
  background: var(--atm-accent); color: #fff; font: inherit; font-weight: 600; cursor: pointer; }
.atm-form button:disabled { opacity: .5; cursor: default; }
.atm-root :focus-visible { outline: 3px solid #1a73e8; outline-offset: 2px; }
.atm-visually-hidden { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
`;

export default AskTheMenu;
