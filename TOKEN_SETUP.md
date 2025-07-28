# 🔐 SAS Authentication Setup

## ⚠️ **Required Setup**

This platform requires SAS OAuth2 authentication. You must generate your own access tokens.

## 🚀 **Token Generation**

1. **Contact your SAS Administrator** for:
   - Client ID and Client Secret
   - OAuth2 endpoint URL
   - SSL certificate files

2. **Generate Tokens** using SAS OAuth2 flow:
   - Visit SAS OAuth2 authorization endpoint
   - Login with your SAS credentials  
   - Exchange authorization code for access/refresh tokens

3. **Configure Authentication** in `src/utils/auth_utils.py`:
   - Update client credentials
   - Set correct SAS environment URLs
   - Configure SSL certificate paths

## 📁 **Required Files**

Create these files locally (they are excluded from Git):
- `access_token.txt` - Your OAuth2 access token
- `refresh_token.txt` - Your OAuth2 refresh token  
- Certificate files in `certificates/` directory

## 📞 **Support**

- **SAS Documentation**: [SAS Viya Authentication](https://documentation.sas.com)
- **SAS Administrator**: For environment-specific setup
- **Technical Support**: For OAuth2 configuration issues

---

**🔒 Keep tokens secure and never commit them to version control!** 