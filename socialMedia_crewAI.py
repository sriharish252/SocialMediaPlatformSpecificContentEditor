import os

from crewai import Agent, Task, Process, Crew
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()   # loads environment variables from .env file

# To load gemini (this api is for free: https://makersuite.google.com/app/apikey)
api_gemini = os.environ.get("GOOGLE_GEMINI_API_KEY") # In .env file paste apikey for the key GOOGLE_GEMINI_API_KEY
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-pro", 
    verbose=True, 
    temperature=0.1, 
    google_api_key=api_gemini
)

def get_modified_content(content: str):
    if content==None or  len(content)<= 0:
        return {}

    # Agent to critize the output of each editor agent to improve the text
    critic_agent = Agent(
        role="Content Critic",
        goal="Criticize the given content to create a better appeal to the target audience",
        backstory="""
                You possess a deep understanding of the content you are critiquing along with its context.
                Grammar & Mechanics: Has a keen eye for grammatical errors, typos, and punctuation mistakes.
                Clarity & Cohesion: Can identify unclear sentences, logical inconsistencies, and a lack of flow within the text.
                Target Audience Awareness: Understands the intended audience and evaluates the content's effectiveness in reaching them.
                Engagement & Impact: Analyzes whether the content is engaging, achieves its intended purpose, and leaves a lasting impression on the reader.
                Constructive Feedback: Delivers criticism in a clear, concise, and actionable manner, offering suggestions for improvement without being overly harsh.
            """,
        verbose=True,  # enable more detailed or extensive output
        allow_delegation=False,  # enable collaboration between agent
        llm=gemini_llm
    )

    # Agent to modify the content to be suitable for Instagram platform
    instagram_agent = Agent(
        role="Instagram Content Editor",
        goal="Modify the user input to make it suitable for Instagram",
        backstory="""You are an expert at curating the content for Instagram, such that it is suitable for a younger generation of target audience.     
            You are good at coming up with ideas on how to appeal to widest possible audience.
            Early adopter & trendsetter: Established a captivating Instagram presence in high school with unique edits and storytelling captions.
            Technical expertise: Became an expert in editing software and mastered the Instagram algorithm during college.
            Community builder: Grew a community around her distinct visual style.
            Brand storytelling: Landed first client (bakery) by demonstrating the ability to capture a brand's essence through photos and captions.
            Adaptability & growth: Now a sought-after editor for diverse clients, constantly honing her skills to stay relevant.
            Empowerment: Passionate about empowering others to effectively use Instagram as a storytelling platform.
            """,
        verbose=True,  # enable more detailed or extensive output
        allow_delegation=False,  # enable collaboration between agent
        llm=gemini_llm
    )

    # Agent to modify the content to be suitable for Tiktok platform
    tiktok_agent = Agent(
        role="Tiktok Content Editor",
        goal="Modify the user input to make it suitable for Tiktok",
        backstory="""You are an intelligent editor, specialized at curating the given content to be suitable for the youngest 
            generation of target audience. This is crucial for marketing to the youth between ages of 13 to 25 years old. 
            Storytelling Ninja: Possesses an innate ability to craft engaging narratives within the short format of TikTok.
            Trend Spotter & Mastermind: Stays ahead of the curve on trending sounds, transitions, and challenges, seamlessly integrating them into edits.
            Editing Virtuoso: Proficient in video editing software and utilizes creative effects, text overlays, and music to enhance storytelling.
            Viral Visionary: Has a keen eye for what resonates with the TikTok audience and understands how to create content that has high viral potential.
            Hook Master: Creates captivating introductions that grab viewers' attention within the first few seconds.
            Efficiency Champion: Can edit videos quickly and efficiently, optimizing workflow to meet tight deadlines.
            """,
        verbose=True,  # enable more detailed or extensive output
        allow_delegation=False,  # enable collaboration between agent
        llm=gemini_llm
    )

    # Agent to modify the content to be suitable for LinkedIn platform
    linkedin_agent = Agent(
        role="LinkedIn Content Editor",
        goal="Modify the user input to make it suitable for LinkedIn",
        backstory="""You are an excellent content writer, capable of modifying the given content for the LinkedIn platform. 
                You create a strong personal brand through compelling written content, showcasing expertise and achievements.
                You craft a perception that highlights professional high quality networks.
                You ensure you content appears in key searches by recruiters, potential employers, prospective collaborators and insdustry professionals.
                You ensure your content adheres to a professional tone and messaging, fostering trust and authority.
                You track and analyze content engagement metrics to tailor content for maximum impact.
            """,
        verbose=True,  # enable more detailed or extensive output
        allow_delegation=False,  # enable collaboration between agent
        llm=gemini_llm
    )

    # Task description for Instagram agent, modify the given content to be suitable for Instagram
    instagram_task = Task(
        description="""
            Modify the given content to be suitable for the Instagram social media platform serving a younger generation of target audience.
            Your content must be engaging and must pertain to the inclusive ideals of modern society.
            It must be engaging with celebrity and trendy references to current events.
            Target Audience: Instagram boasts a diverse user base, but skews heavily towards younger demographics (16-34 year olds). This audience generally craves engaging and visually appealing content.
            Desired Modifications:
            Length: Condense the content into a concise and impactful format suitable for Instagram's character limitations (typically captions under 2,200 characters).
            Engagement: Rephrase the content to be more conversational and interactive. Consider incorporating questions or calls to action to encourage user comments and discussions.
            Hashtags: Suggest relevant and trending hashtags that will increase the content's discoverability on Instagram.
        """ + "Given Content:" + content,
        agent= instagram_agent,
        expected_output = "Modified content optimized for the Instagram platform.",
    )

    # Task description for Tiktok agent
    tiktok_task = Task(
        description="""Modify the given content to make it suitable for Tiktok.
            Curate the given content to be suitable for the youngest generation of target audience. 
            Your primary marketing should be to the youth between ages of 13 to 22 years old. 
            Make sure to make it appropriate for the Tiktok platform, yet retain the original meaning of the given content.
            Your content must be concise, quick to the point, funny and entertaining to read. 
            Incorporate modern youth slang and trendy refenrences.
            Target Audience: TikTok's primary audience falls within Gen Z (born between 1997-2012) and late Millennials (born between 1990-1996). 
            This audience thrives on short-form, entertaining content with a strong emphasis on music, humor, and trending challenges.

            Desired Modifications:
            Length: Condense the content into a highly engaging and informative format suitable for TikTok's short video limitations (typically 15-60 seconds).
            Hook & Storytelling: Craft a captivating introduction that grabs viewers within the first few seconds and utilize creative storytelling techniques to convey the core message in a digestible way.
            Challenge or Trend Integration: Explore opportunities to integrate the content with a trending TikTok challenge or hashtag to increase discoverability and user participation.
        """ + "Given Content:" + content,
        agent= tiktok_agent,
        expected_output = "Modified content optimized for the Tiktok platform.",
    )

    # Task description for LinkedIn agent
    linkedin_task = Task(
        description="""Modify the given content to make it suitable for the LinkedIn social media platform.
            Curate the given content to be suitable for professionals. 
            This is crucial for marketing to other businesses and create potential collaborations and brand partnerships. 
            Your target audience is between the ages of 30 to 55 years old. 
            Target Audience: LinkedIn is a professional networking platform attracting individuals across various industries and career stages. 
            Users seek to establish themselves as thought leaders, connect with potential employers or clients, and stay updated on industry trends.

            Desired Modifications:
            Professional Tone: Ensure the content adopts a professional and informative writing style, omitting overly casual language or humor.
            Credibility Boost: Identify sections within the content that could be strengthened by incorporating relevant data, statistics, or expert quotes to bolster its credibility.
            Actionable Insights: Repurpose the content to offer actionable insights or takeaways valuable to the LinkedIn audience in their professional endeavors.
            Keywords & Hashtags: Suggest relevant industry-specific keywords and trending LinkedIn hashtags to increase the content's discoverability for your target audience.
            Call to Action: Craft a call to action that encourages user engagement, such as inviting discussions in the comments section, sharing their experiences, or following your profile for future content.
        """ + "Given Content:" + content,
        agent= linkedin_agent,
        expected_output = "Modified content optimized for the LinkedIn platform.",
    )

    # Task description for Critic agent, provide feedback to agent
    critic_task = Task(
        description="""
            Provide constructive criticism and feedback on the content modified by another LLM agent. Appreciate the positives and point out the flaws.
            Platform Suitability: Analyze the content's overall suitability for the chosen social media platform. Consider factors like target audience, platform-specific trends, and optimal content length.
            Clarity & Concision: Identify areas where the content could be clearer or more concise. Suggest edits to improve readability and ensure the message is conveyed effectively within the platform's character limitations (if applicable).
            Spelling & Grammar: Locate any spelling mistakes, grammatical errors, or punctuation issues that hinder understanding.
            Engagement & Impact: Assess whether the content is engaging for the target audience. Suggest revisions to make it more interactive, spark conversation, or capture attention.
            Call to Action: Recommend incorporating a clear call to action that encourages user interaction, such as liking, commenting, sharing, or following your account.

            Always paste the given input content at the end of the feedback with the title 'Given Content:'.
        """,
        agent= critic_agent,
        expected_output = "Contructive criticism and feedback on the content modified by another LLM agent.",
    )

    # Rewrite Task description for Instagram agent, to modify content based on feedback
    instagram_rewrite_task = Task(
        description="""Alter the content based on the feedback provided. 
        """,
        agent= instagram_agent,
        expected_output = "Modified content optimized for Instagram",
    )

    # Rewrite Task description for Tiktok agent
    tiktok_rewrite_task = Task(
        description="""Alter the content based on the feedback provided. 
        """,
        agent= tiktok_agent,
        expected_output = "Modified content optimized for Tiktok",
    )

    # Rewrite Task description for LinkedIn agent
    linkedin_rewrite_task = Task(
        description="""Alter the content based on the feedback provided. 
        """,
        agent= linkedin_agent,
        expected_output = "Modified content optimized for LinkedIn",
    )

    # Crew definition with agents and tasks specific to Instagram.
    # Workflow: 
    #       instagram_agent performs instagram_task passes modified content to -> critic_agent, 
    #       critic_agent then performs critic_task passes feedback to -> instagram_agent, 
    #       instagram_agent then performs instagram_rewrite_task
    instagram_crew = Crew(
        agents=[instagram_agent, critic_agent, instagram_agent],
        tasks=[instagram_task, critic_task, instagram_rewrite_task],
        verbose=False,  # False disables logging to console
        process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
    )

    # Crew definition with agents and tasks specific to Tiktok.
    tiktok_crew = Crew(
        agents=[tiktok_agent, critic_agent, tiktok_agent],
        tasks=[tiktok_task, critic_task, tiktok_rewrite_task],
        verbose=False,
        process=Process.sequential,
    )

    # Crew definition with agents and tasks specific to LinkedIn.
    linkedin_crew = Crew(
        agents=[linkedin_agent, critic_agent, linkedin_agent],
        tasks=[linkedin_task, critic_task, linkedin_rewrite_task],
        verbose=False,
        process=Process.sequential,
    )

    # Trigger each crew to work
    instagram_content = instagram_crew.kickoff()
    tiktok_content = tiktok_crew.kickoff()
    linkedin_content = linkedin_crew.kickoff()

    # Return modified content as dictionary
    return {'instagram_content': instagram_content, 'tiktok_content': tiktok_content, 'linkedin_content': linkedin_content}

