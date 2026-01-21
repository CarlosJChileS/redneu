import { Router, Request, Response } from 'express'

const router = Router()

const GROQ_ENDPOINT = 'https://api.groq.com/openai/v1/chat/completions'
// Modelo de TEXTO (no visión)
const GROQ_MODEL = 'llama-3.3-70b-versatile'

interface AnalyzeRequest {
  description: string
}

router.post('/analyze', async (req: Request, res: Response) => {
  const GROQ_API_KEY = process.env.GROQ_API_KEY

  try {
    const { description } = req.body as AnalyzeRequest

    if (!description) {
      return res.status(400).json({ success: false, error: 'No description' })
    }

    if (!GROQ_API_KEY) {
      return res.status(500).json({ success: false, error: 'No API key' })
    }

    console.log('📡 Llamando a Groq (texto)...')
    console.log('📝 Descripción:', description.substring(0, 100) + '...')

    const response = await fetch(GROQ_ENDPOINT, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${GROQ_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: GROQ_MODEL,
        messages: [
          {
            role: 'system',
            content: `You are an expert at recognizing handwritten digits (0-9) based on their visual characteristics description. 
Analyze the description of a handwritten digit and determine which digit (0-9) it most likely represents.
Reply with ONLY a JSON object in this exact format: {"digit": X, "confidence": Y}
Where X is the digit (0-9) and Y is your confidence (0.0 to 1.0).
Do not include any other text, just the JSON.`
          },
          {
            role: 'user',
            content: `Based on this description of a handwritten digit, what digit (0-9) is it?\n\n${description}`
          }
        ],
        temperature: 0.1,
        max_tokens: 50
      })
    })

    if (!response.ok) {
      const errorData = await response.text()
      console.error('❌ Groq error:', response.status, errorData)
      return res.status(response.status).json({ 
        success: false, 
        error: `Groq: ${response.status}`
      })
    }

    const data = await response.json() as { choices?: { message?: { content?: string } }[] }
    const content = data.choices?.[0]?.message?.content?.trim()
    console.log('📝 Groq responde:', content)

    // Intentar parsear JSON
    try {
      const parsed = JSON.parse(content || '')
      if (typeof parsed.digit === 'number' && parsed.digit >= 0 && parsed.digit <= 9) {
        console.log(`✅ Groq predice: ${parsed.digit} (${(parsed.confidence * 100).toFixed(0)}%)`)
        return res.json({ 
          success: true, 
          digit: parsed.digit, 
          confidence: parsed.confidence || 0.8 
        })
      }
    } catch {
      // Si no es JSON, buscar un dígito en la respuesta
      const match = content?.match(/\d/)
      if (match) {
        const digit = parseInt(match[0])
        console.log(`✅ Groq predice: ${digit}`)
        return res.json({ success: true, digit, confidence: 0.7 })
      }
    }

    return res.json({ success: false, error: 'No digit found', raw: content })

  } catch (error) {
    console.error('❌ Error:', error)
    return res.status(500).json({ success: false, error: 'Server error' })
  }
})

router.get('/status', (_, res) => {
  res.json({ enabled: !!process.env.GROQ_API_KEY, model: GROQ_MODEL })
})

export { router as groqRouter }
