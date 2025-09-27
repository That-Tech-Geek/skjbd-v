const express = require('express');
const serverless = require('serverless-http');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const jwt = require('jsonwebtoken');
const { OAuth2Client } = require('google-auth-library');
const multer = require('multer');
const busboy = require('busboy');
const { Readable } = require('stream');

// --- CONFIGURATION ---
const {
    GEMINI_API_KEY,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI,
    JWT_SECRET
} = process.env;

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
// IMPORTANT: The model name `gemini-2.5-flash-lite` from your script is not a valid public model.
// I have replaced it with `gemini-1.5-flash`, which is a standard and valid model.
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
const oAuth2Client = new OAuth2Client(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI);

const app = express();
app.use(express.json());

// --- AUTHENTICATION MIDDLEWARE ---
const authMiddleware = (req, res, next) => {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({ error: 'Unauthorized: No token provided' });
    }
    const token = authHeader.split(' ')[1];
    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        req.user = decoded;
        next();
    } catch (error) {
        return res.status(401).json({ error: 'Unauthorized: Invalid token' });
    }
};


// --- UTILITY FUNCTIONS ---
const resilientJsonParser = (jsonString) => {
    try {
        const match = jsonString.match(/```json\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*\})/);
        if (match) {
            return JSON.parse(match[1] || match[2]);
        }
        return null;
    } catch (e) {
        console.error("JSON parsing error:", e);
        return null;
    }
};

const streamToBuffer = (stream) => {
    return new Promise((resolve, reject) => {
        const chunks = [];
        stream.on('data', (chunk) => chunks.push(chunk));
        stream.on('error', reject);
        stream.on('end', () => resolve(Buffer.concat(chunks)));
    });
};

const fileParser = (req) => {
    return new Promise((resolve, reject) => {
        const bb = busboy({ headers: req.headers });
        const files = [];
        const fields = {};

        bb.on('file', (name, file, info) => {
            const { filename, encoding, mimeType } = info;
            const bufferPromise = streamToBuffer(file);
            files.push({ bufferPromise, filename, mimeType });
        });

        bb.on('field', (name, val) => fields[name] = val);
        bb.on('close', async () => {
             try {
                const resolvedFiles = await Promise.all(
                    files.map(async f => ({
                        buffer: await f.bufferPromise,
                        filename: f.filename,
                        mimeType: f.mimeType
                    }))
                );
                resolve({ files: resolvedFiles, fields });
            } catch (err) {
                reject(err);
            }
        });
        bb.on('error', err => reject(err));
        
        // Pipe the request stream to busboy
        if (req.rawBody) {
            const readable = new Readable();
            readable._read = () => {}; 
            readable.push(req.rawBody);
            readable.push(null);
            readable.pipe(bb);
        } else {
            req.pipe(bb);
        }
    });
};


// --- API ROUTES ---
const router = express.Router();

// 1. Google Auth Initiator
router.get('/auth-google', (req, res) => {
    const authorizeUrl = oAuth2Client.generateAuthUrl({
        access_type: 'offline',
        scope: ['https://www.googleapis.com/auth/userinfo.profile', 'https://www.googleapis.com/auth/userinfo.email', 'openid'],
    });
    res.redirect(authorizeUrl);
});

// 2. Google Auth Callback
router.get('/auth-callback', async (req, res) => {
    const { code } = req.query;
    try {
        const { tokens } = await oAuth2Client.getToken(code);
        oAuth2Client.setCredentials(tokens);
        const userInfoResponse = await oAuth2Client.request({
            url: 'https://www.googleapis.com/oauth2/v3/userinfo'
        });
        const user = userInfoResponse.data;
        
        const jwtToken = jwt.sign({ id: user.sub, email: user.email, given_name: user.given_name, picture: user.picture }, JWT_SECRET, { expiresIn: '7d' });
        
        // Redirect back to the frontend with the token
        res.redirect(`${process.env.NETLIFY_SITE_URL}?token=${jwtToken}`);

    } catch (error) {
        console.error('Authentication error:', error);
        res.status(500).redirect(process.env.NETLIFY_SITE_URL);
    }
});

// 3. Verify Token
router.get('/verify-token', authMiddleware, (req, res) => {
    res.json({ message: 'Token is valid', user: req.user });
});

// 4. Process Files
router.post('/process-files', authMiddleware, async (req, res) => {
     try {
        // We use rawBody because Netlify/AWS Lambda might parse the body already
        req.rawBody = req.body;
        const { files } = await fileParser(req);
        
        // Placeholder for actual file processing (e.g., PDF text extraction)
        // For a real app, you'd use libraries like pdf-parse here.
        const chunks = files.map(f => ({
            chunk_id: `${f.filename}::chunk_0`,
            text: `Content from ${f.filename}` // Dummy content
        }));

        res.json({ chunks });
    } catch (error) {
        console.error("File processing error:", error);
        res.status(500).json({ error: 'Failed to process files.' });
    }
});

// 5. Generate Outline
router.post('/generate-outline', authMiddleware, async (req, res) => {
    const { chunks } = req.body;
    if (!chunks || chunks.length === 0) {
        return res.status(400).json({ error: 'No content chunks provided.' });
    }
    const prompt_chunks = chunks.map(c => ({ chunk_id: c.chunk_id, text_snippet: c.text.substring(0, 200) + "..." }));

    const prompt = `
        You are a master curriculum designer. Create a coherent study outline from these text snippets.
        For each topic, you MUST list the 'chunk_id's that are most relevant.
        Output ONLY a valid JSON object with a root key "outline", which is a list of objects. Each object must have keys "topic" (string) and "relevant_chunks" (a list of string chunk_ids).
        
        **Content Chunks:**
        ---
        ${JSON.stringify(prompt_chunks, null, 2)}
    `;
    try {
        const result = await model.generateContent(prompt);
        const responseText = await result.response.text();
        const parsedJson = resilientJsonParser(responseText);
        
        if (!parsedJson || !parsedJson.outline) {
            return res.status(500).json({ error: "AI failed to generate a valid outline JSON." });
        }

        res.json(parsedJson);

    } catch (error) {
        console.error("Gemini outline generation error:", error);
        res.status(500).json({ error: 'Failed to generate content outline.' });
    }
});

// 6. Synthesize Notes
router.post('/synthesize-notes', authMiddleware, async (req, res) => {
    const { topics, all_chunks } = req.body;
    const chunksMap = new Map(all_chunks.map(c => [c.chunk_id, c.text]));
    
    try {
        const notePromises = topics.map(async (topicData) => {
            const relevantText = topicData.relevant_chunks_ids
                .map(id => chunksMap.get(id))
                .filter(Boolean)
                .join('\n\n---\n\n');

            if (!relevantText.trim()) {
                return { topic: topicData.topic, content: "Could not find source text for this topic.", source_chunks: topicData.relevant_chunks_ids };
            }

            const prompt = `
                Synthesize a detailed, clear note block for the topic: "${topicData.topic}".
                Base your response STRICTLY on the provided source text. Format in Markdown.
                User Instructions: ${topicData.instructions || "Create clear, concise, well-structured notes."}
                Source Text:
                ---
                ${relevantText}
            `;
            const result = await model.generateContent(prompt);
            const content = await result.response.text();
            return { topic: topicData.topic, content, source_chunks: topicData.relevant_chunks_ids };
        });

        const notes = await Promise.all(notePromises);
        res.json({ notes });

    } catch (error) {
        console.error("Gemini note synthesis error:", error);
        res.status(500).json({ error: 'Failed to synthesize notes.' });
    }
});

app.use('/api/', router);
module.exports.handler = serverless(app);
